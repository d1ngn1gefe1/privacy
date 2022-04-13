import inspect

import torch.distributed
from opacus import PrivacyEngine
from opacus.data_loader import DPDataLoader
from opacus.privacy_engine import forbid_accumulation_hook
from pytorch_lightning.callbacks.base import Callback
from types import MethodType


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(delta=self.cfg.delta)
  # same across all devices, no need to sync
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=False, prog_bar=True)


def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()

  # old optimizer and lr scheduler
  dict_optimizers = self.configure_optimizers_old()
  optimizer_old, scheduler_old = dict_optimizers['optimizer'], dict_optimizers['lr_scheduler']

  # new optimizer
  distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
  expected_batch_size = int(len(dataloader.dataset)/len(dataloader)/len(self.cfg.gpus))
  optimizer = self.privacy_engine._prepare_optimizer(optimizer_old,
                                                     distributed=distributed,
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=self.cfg.c,
                                                     expected_batch_size=expected_batch_size,
                                                     clipping='flat')
  sample_rate = 1/len(dataloader)
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

  # new lr scheduler
  kwargs = {key:scheduler_old.__dict__[key]
            for key in inspect.signature(scheduler_old.__class__.__init__).parameters.keys()
            if key not in ['self', 'optimizer']}
  scheduler = scheduler_old.__class__(optimizer, **kwargs)

  print(f'Net: {type(self.net)}, Optimizer: {type(optimizer)}')
  return {'optimizer': optimizer, 'lr_scheduler': scheduler}


def train_dataloader(self):
  distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
  dataloader = DPDataLoader.from_data_loader(self.train_dataloader_old(), distributed=distributed)
  return dataloader


class DPCallback(Callback):
  def __init__(self):
    pass

  def setup(self, trainer, pl_module, stage):
    pl_module.privacy_engine = PrivacyEngine()
    pl_module.on_train_epoch_end = MethodType(on_train_epoch_end, pl_module)

    # make net private
    pl_module.net = pl_module.privacy_engine._prepare_model(pl_module.net)
    # pl_module.net.register_forward_pre_hook(forbid_accumulation_hook)  # TODO

    # make dataloader private
    trainer.datamodule.train_dataloader_old = trainer.datamodule.train_dataloader
    trainer.datamodule.train_dataloader = MethodType(train_dataloader, trainer.datamodule)

    # make optimizer private
    pl_module.configure_optimizers_old = pl_module.configure_optimizers
    pl_module.configure_optimizers = MethodType(configure_optimizers, pl_module)