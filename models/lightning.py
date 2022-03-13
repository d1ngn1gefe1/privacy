from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
import pytorch_lightning as pl
import torch


class DPLightningModule(pl.LightningModule):
  def __init__(self, module):
    super().__init__()
    self.module = module
    self.cfg = module.cfg
    self.privacy_engine = PrivacyEngine()

  def forward(self, x):
    return self.module.forward(x)

  def training_step(self, batch, batch_idx):
    return self.module.training_step(batch, batch_idx)

  def validation_step(self, batch, batch_idx):
    return self.module.validation_step(batch, batch_idx)

  def configure_optimizers(self):
    optimizer = self.module.configure_optimizers()

    data_loader = (self.trainer._data_connector._train_dataloader_source.dataloader())

    net_dp, optimizer, dataloader = self.privacy_engine.make_private(
      module=self,
      optimizer=optimizer,
      data_loader=data_loader,
      noise_multiplier=self.cfg.sigma,
      max_grad_norm=self.cfg.c,
      poisson_sampling=isinstance(data_loader, DPDataLoader),
    )
    self.dp = {'model': net_dp}

    return optimizer

  def on_train_epoch_end(self):
    epsilon = self.privacy_engine.get_epsilon(self.cfg.delta)
    self.module.log('epsilon', epsilon, on_epoch=True, sync_dist=True, prog_bar=True)

