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
    self.net = get_net(cfg.net, cfg.num_classes, cfg.mode != 'from_scratch')
    if cfg.mode == 'linear_probing':
      for param in self.net.parameters():
        param.requires_grad = False
      for param in self.net.get_classifier().parameters():
        param.requires_grad = True
    self.top1 = torchmetrics.Accuracy(top_k=1)

  def forward(self, x):
    return self.net(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)
    loss = F.cross_entropy(y_hat, y)
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, y)

    self.log('train/loss', loss, prog_bar=True)
    self.log('train/acc1', top1, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.net(x)
    loss = F.cross_entropy(y_hat, y)
    pred = F.softmax(y_hat, dim=-1)
    top1 = self.top1(pred, y)

    self.log('val/loss', loss, sync_dist=True, prog_bar=True)
    self.log('val/acc1', top1, sync_dist=True, prog_bar=True)

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
