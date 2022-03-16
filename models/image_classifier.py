from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

from .nets import get_net


class ImageClassifierModule(LightningModule):
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
    
    # multi-label vs single-label
    if cfg.dataset == 'chexpert':
      self.metric = torchmetrics.AUROC(cfg.num_classes)
      self.name_stat = 'roc'
      self.get_loss = F.multilabel_soft_margin_loss
    else: 
      self.metric = torchmetrics.Accuracy(top_k=1)
      self.name_stat = 'acc1'
      self.get_loss = F.cross_entropy

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)
    loss = self.get_loss(y_hat, y)
    pred = F.softmax(y_hat, dim=-1)
    stat = self.metric(pred, y)

    self.log('train/loss', loss, prog_bar=True)
    self.log(f'train/{self.name_stat}', stat, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)
    loss = self.get_loss(y_hat, y)   
    pred = F.softmax(y_hat, dim=-1)
    stat = self.metric(pred, y)

    self.log('val/loss', loss, sync_dist=True, prog_bar=True)
    self.log(f'val/{self.name_stat}', stat, sync_dist=True, prog_bar=True)

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
