from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
import pytorch_lightning as pl


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
    # Reference: https://github.com/pytorch/opacus/blob/main/examples/mnist_lightning.py#L98
    optimizer = self.module.configure_optimizers()

    data_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
    if hasattr(self, 'dp'):
      self.dp['module'].remove_hooks()
    module, optimizer, data_loader = self.privacy_engine.make_private(
      module=self.module,
      optimizer=optimizer,
      data_loader=data_loader,
      noise_multiplier=self.cfg.sigma,
      max_grad_norm=self.cfg.c,
      poisson_sampling=isinstance(data_loader, DPDataLoader),
    )
    self.dp = {'module': module}
    return optimizer

  def on_train_epoch_end(self):
    epsilon = self.privacy_engine.get_epsilon(self.cfg.delta)
    self.log('epsilon', epsilon, on_epoch=True, sync_dist=True, prog_bar=True)
