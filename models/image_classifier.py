import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
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

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)

    loss = self.get_loss(y_hat, y)
    self.log('train/loss', loss, prog_bar=True)

    pred = self.get_pred(y_hat)
    for name, get_stat in self.metrics.items():
      self.log(f'train/{name}', get_stat(pred, y), prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)

    loss = self.get_loss(y_hat, y)
    self.log('val/loss', loss, sync_dist=True, prog_bar=True)

    pred = self.get_pred(y_hat)
    for name, get_stat in self.metrics.items():
      stat = get_stat(pred, y)
      self.log(f'val/{name}', stat, sync_dist=True, prog_bar=True)

      if self.cfg.dp:
        epsilon = self.privacy_engine.get_epsilon(self.cfg.delta)
        self.log(f'val/{name}-div-log_epsilon', stat/np.log(epsilon), sync_dist=True, prog_bar=True)

  def test_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)

    loss = self.get_loss(y_hat, y)
    self.log('test/loss', loss, sync_dist=True, prog_bar=True)

    pred = self.get_pred(y_hat)
    for name, get_stat in self.metrics.items():
      stat = get_stat(pred, y)
      self.log(f'test/{name}', stat, sync_dist=True, prog_bar=True)

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
