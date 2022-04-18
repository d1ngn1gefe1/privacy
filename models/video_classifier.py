from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD
import torchmetrics

from .nets import get_net
import utils


class VideoClassifierModule(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.net = get_net(cfg.net, cfg.num_classes, cfg.mode != 'from_scratch', cfg.dir_weights)

    # set trainable parameters
    if cfg.mode == 'linear_probing' or cfg.mode == 'adapter':
      for param in self.net.parameters():
        param.requires_grad = False
      for param in self.net.get_classifier().parameters():
        param.requires_grad = True

    # set task-specific variables
    if cfg.task == 'multi-class':
      self.get_loss = F.cross_entropy
      self.get_pred = lambda x: F.softmax(x, dim=-1)
      self.metrics = nn.ModuleDict({'acc': torchmetrics.Accuracy(average='micro')})
    elif cfg.task == 'multi-label':
      self.get_loss = F.multilabel_soft_margin_loss
      self.get_pred = torch.sigmoid
      self.metrics = nn.ModuleDict({'acc': torchmetrics.Accuracy(average='macro', num_classes=cfg.num_classes),
                                    'roc': torchmetrics.AUROC(num_classes=cfg.num_classes)})
    else:
      raise NotImplementedError

    self.ensemble_method = 'sum'  # sum or max

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    # TODO: remove print
    # utils.info(self.current_epoch, self.global_step, batch_idx, y[:5])
    y_hat = self.net(x)

    loss = self.get_loss(y_hat, y)
    self.log('train/loss', loss, batch_size=batch_size, prog_bar=True)

    pred = self.get_pred(y_hat)
    for name, get_stat in self.metrics.items():
      self.log(f'train/{name}', get_stat(pred, y), batch_size=batch_size, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    y_hat = self.net(x)

    loss = self.get_loss(y_hat, y)
    self.log('val/loss', loss, batch_size=batch_size, sync_dist=True, prog_bar=True)

    pred = self.get_pred(y_hat)
    #for name, get_stat in self.metrics.items():
    #  self.log(f'val/{name}', get_stat(pred, y), batch_size=batch_size, sync_dist=True, prog_bar=True)

    video_ids = batch['video_index'].clone()
    self._ensemble_at_video_level(pred, y, video_ids)

  def test_step(self, batch, batch_idx):
    batch_size = batch['video'][0].shape[0] if isinstance(batch['video'], list) else batch['video'].shape[0]
    x, y = batch['video'], batch['label']
    y_hat = self.net(x)

    loss = self.get_loss(y_hat, y)
    self.log('test/loss', loss, batch_size=batch_size, sync_dist=True, prog_bar=True)

    pred = self.get_pred(y_hat)
    video_ids = batch['video_index'].clone()
    self._ensemble_at_video_level(pred, y, video_ids)

  def predict_step(self, batch, batch_idx):
    pass

  def on_train_epoch_start(self):
    """ Needed for shuffling in distributed training
    Reference:
     - https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/train.py#L96
     - https://pytorch.org/docs/master/data.html#torch.utils.data.distributed.DistributedSampler
    """
    if utils.is_ddp():
      self.trainer.datamodule.dataset_train.dataset.video_sampler.set_epoch(self.trainer.current_epoch)

    self._prepare_ensemble()

  def on_validation_epoch_end(self):
    self._collect_results('val')
    self._prepare_ensemble()

  def on_test_epoch_end(self):
    self._collect_results('test')
    self._prepare_ensemble()

  def _prepare_ensemble(self):
    # These are used for data ensembling in the test stage.
    self.video_preds = {}
    self.video_labels = {}
    self.video_clips_cnts = {}

  def _collect_results(self, phase='val'):
    for video_id in self.video_preds:
      self.video_preds[video_id] = (
        self.video_preds[video_id] / self.video_clips_cnts[video_id]
      )
      video_preds = torch.stack(list(self.video_preds.values()), dim=0)
      video_labels = torch.tensor(
        list(self.video_labels.values()),
        device=self.video_labels[video_id].device,
      )
    for name, get_stat in self.metrics.items():
      self.log(f'{phase}/{name}', get_stat(video_preds, video_labels), on_epoch=True, prog_bar=True)

  def _ensemble_at_video_level(self, preds, labels, video_ids):
    for i in range(preds.shape[0]):
      vid_id = int(video_ids[i])
      self.video_labels[vid_id] = labels[i]
      if vid_id not in self.video_preds:
        self.video_preds[vid_id] = torch.zeros(
          (self.cfg.num_classes), device=preds.device, dtype=preds.dtype
        )
        self.video_clips_cnts[vid_id] = 0

      if self.ensemble_method == 'sum':
        self.video_preds[vid_id] += preds[i]
      elif self.ensemble_method == 'max':
        self.video_preds[vid_id] = torch.max(self.video_preds[vid_id], preds[i])
      self.video_clips_cnts[vid_id] += 1

  def configure_optimizers(self):
    optimizer = SGD(self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.num_epochs)
    return {'optimizer': optimizer, 'lr_scheduler': scheduler}

  def optimizer_step(
      self,
      epoch,
      batch_idx,
      optimizer,
      optimizer_idx,
      optimizer_closure,
      on_tpu=False,
      using_native_amp=False,
      using_lbfgs=False
  ):
    # linear warmup
    if self.trainer.global_step < self.cfg.warmup_steps:
      lr_scale = min(1.0, float(self.trainer.global_step+1)/self.cfg.warmup_steps)
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale*self.cfg.lr

    optimizer.step(closure=optimizer_closure)
