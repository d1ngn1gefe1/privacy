import inspect

from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.privacy_engine import forbid_accumulation_hook
from pytorch_lightning.callbacks.base import Callback
from torch.utils.data import DataLoader
from types import MethodType

import utils
from data.datasets.subsample_dataset import SubsampleDataset


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(delta=self.cfg.delta)
  # same across all devices, no need to sync
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=False, prog_bar=True)


def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()

  # old optimizer and lr scheduler
  optimizer_old, scheduler_old = self.configure_optimizers_old()
  optimizer_old, scheduler_old = optimizer_old[0], scheduler_old[0]

  # new optimizer
  len_dataloader = len(dataloader[0]) if isinstance(dataloader, list) else len(dataloader)
  len_dataset = len(dataloader[0].dataset) if isinstance(dataloader, list) else len(dataloader.dataset)
  expected_batch_size = int(len_dataset/len_dataloader/len(self.cfg.gpus))
  optimizer = self.privacy_engine._prepare_optimizer(optimizer_old,
                                                     distributed=utils.is_ddp(),
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=self.cfg.c,
                                                     expected_batch_size=expected_batch_size,
                                                     clipping='flat')
  sample_rate = 1/len_dataloader
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

  # new lr scheduler
  kwargs = {key:scheduler_old.__dict__[key]
            for key in inspect.signature(scheduler_old.__class__.__init__).parameters.keys()
            if key not in ['self', 'optimizer']}
  scheduler = scheduler_old.__class__(optimizer, **kwargs)

  if isinstance(dataloader, list):
    return [optimizer, optimizer_old], [scheduler, scheduler_old]
  else:
    return [optimizer], [scheduler]


def training_step(self, batch, batch_idx, optimizer_idx):
  x, y = batch[optimizer_idx]
  y_hat = self(x)

  loss = self.get_loss(y_hat, y)
  self.log(f'train/loss{optimizer_idx}', loss, prog_bar=True)

  pred = self.get_pred(y_hat)
  for name, get_stat in self.metrics.items():
    self.log(f'train/{name}{optimizer_idx}', get_stat(pred, y), prog_bar=True)

  return loss


def train_dataloader(self):
  dataloader = DPDataLoader.from_data_loader(self.train_dataloader_old(), distributed=utils.is_ddp())

  if 'ratio_public' in self.cfg:
    dataset_subsample = SubsampleDataset(self.dataset_train, self.cfg['ratio_public'])
    # TODO: cleanup
    dataloader_subsample = DataLoader(dataset_subsample, batch_size=self.cfg.batch_size//len(self.cfg.gpus),
                                      shuffle=True, num_workers=self.cfg.num_workers, pin_memory=True)
    return [dataloader, dataloader_subsample]
  else:
    return dataloader


class DPCallback(Callback):
  def __init__(self):
    pass

  def setup(self, trainer, pl_module, stage):
    pl_module.privacy_engine = PrivacyEngine()
    pl_module.on_train_epoch_end = MethodType(on_train_epoch_end, pl_module)
    if 'ratio_public' in trainer.datamodule.cfg:
      pl_module.training_step = MethodType(training_step, pl_module)

    # make net private
    pl_module.net = pl_module.privacy_engine._prepare_model(pl_module.net)
    # pl_module.net.register_forward_pre_hook(forbid_accumulation_hook)  # TODO

    # make dataloader private
    trainer.datamodule.train_dataloader_old = trainer.datamodule.train_dataloader
    trainer.datamodule.train_dataloader = MethodType(train_dataloader, trainer.datamodule)

    # make optimizer private
    pl_module.configure_optimizers_old = pl_module.configure_optimizers
    pl_module.configure_optimizers = MethodType(configure_optimizers, pl_module)
