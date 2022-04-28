from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchmetrics

from .nets import get_net


class BaseClassifierModule(LightningModule):
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

  def forward(self, x):
    return self.net(x)

  def configure_optimizers(self):
    if self.cfg.optimizer == 'sgd':
      optimizer = SGD(self.net.parameters(), lr=self.cfg.lr, momentum=self.cfg.momentum, weight_decay=self.cfg.wd)
    elif self.cfg.optimizer == 'adamw':
      optimizer = AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wd)
    else:
      raise NotImplementedError
    scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg.num_epochs)
    return [optimizer], [scheduler]

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
    optimizer.step(closure=optimizer_closure)

    # linear warmup
    if self.trainer.global_step < self.cfg.warmup_steps:
      lr_scale = min(1.0, float(self.trainer.global_step+1)/self.cfg.warmup_steps)
      for pg in optimizer.param_groups:
        pg['lr'] = lr_scale*self.cfg.lr
