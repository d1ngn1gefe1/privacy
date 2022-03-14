from pytorch_lightning import LightningModule
import torch.nn.functional as F
from torch.optim import SGD
import torchmetrics

from .nets import get_net


class ImageClassifierModule(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.net = get_net(cfg)
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
    return optimizer
