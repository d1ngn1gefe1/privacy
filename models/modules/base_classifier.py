from pytorch_lightning import LightningModule
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.nets import get_net
from .misc import set_mode, set_task


class BaseClassifierModule(LightningModule):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.net = get_net(cfg)

    set_mode(self.net, cfg)  # set trainable parameters
    self.get_loss, self.get_pred, self.metrics_train, self.metrics_val, self.metrics_test = set_task(cfg)  # set task

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
