from functools import partial
import inspect
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
from opacus.privacy_engine import forbid_accumulation_hook
from opacus.utils.batch_memory_manager import BatchSplittingSampler, wrap_data_loader
from opacus.utils.module_utils import trainable_modules
from pytorch_lightning.callbacks.base import Callback
from torch.utils.data import DataLoader
from types import MethodType

import utils


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(delta=self.cfg.delta)
  # same across all devices, no need to sync
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=False, prog_bar=True)


def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()

  expected_batch_size = int(len(dataloader.dataset)/len(dataloader)/len(self.cfg.gpus))
  sample_rate = 1/len(dataloader)

  # get privacy budgets
  assert sum([hasattr(self.cfg, 'sigma'), hasattr(self.cfg, 'epsilon')]) == 1
  if hasattr(self.cfg, 'epsilon'):
    self.cfg.sigma = get_noise_multiplier(target_epsilon=self.cfg.epsilon, target_delta=self.cfg.delta,
                                          sample_rate=sample_rate, epochs=self.cfg.num_epochs,
                                          accountant=self.privacy_engine.accountant.mechanism())

  # old optimizer and lr scheduler
  optimizer_old, scheduler_old = self.configure_optimizers_old()
  optimizer_old, scheduler_old = optimizer_old[0], scheduler_old[0]

  # new optimizer
  optimizer = self.privacy_engine._prepare_optimizer(optimizer_old,
                                                     distributed=utils.is_ddp(),
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=self.cfg.c,
                                                     expected_batch_size=expected_batch_size,
                                                     clipping='flat')
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

  # new lr scheduler
  kwargs = {key:scheduler_old.__dict__[key]
            for key in inspect.signature(scheduler_old.__class__.__init__).parameters.keys()
            if key not in ['self', 'optimizer']}
  scheduler = scheduler_old.__class__(optimizer, **kwargs)

  return [optimizer], [scheduler]


def train_dataloader(self):
  dataloader = DPDataLoader.from_data_loader(self.train_dataloader_old(), distributed=utils.is_ddp())

  # batch memory manager
  if hasattr(self.cfg, 'max_batch_size') and len(self.trainer.optimizers) > 0:
    delattr(BatchSplittingSampler, '__len__')
    dataloader = wrap_data_loader(data_loader=dataloader,
                                  max_batch_size=self.cfg.max_batch_size,
                                  optimizer=self.trainer.optimizers[0])

  return dataloader


class DPCallback(Callback):
  def setup(self, trainer, pl_module, stage):
    pl_module.privacy_engine = PrivacyEngine()
    pl_module.on_train_epoch_end = MethodType(on_train_epoch_end, pl_module)

    # make net private
    pl_module.net = pl_module.privacy_engine._prepare_model(pl_module.net)
    pl_module.register_full_backward_hook(forbid_accumulation_hook)

    # change batch dimension in CLIP ViT
    if trainer.datamodule.cfg.net == 'vit' and trainer.datamodule.cfg.weight == 'pretrain_clip':
      for _module_name, module in trainable_modules(pl_module.net._module):
        if '.attn.' in _module_name:
          key = next(iter(module._backward_hooks.keys()))
          hook = module._backward_hooks[key]
          module._backward_hooks[key] = partial(pl_module.net.capture_backprops_hook,
                                                loss_reduction=hook.keywords['loss_reduction'],
                                                batch_first=False)

    # make dataloader private
    trainer.datamodule.train_dataloader_old = trainer.datamodule.train_dataloader
    trainer.datamodule.train_dataloader = MethodType(train_dataloader, trainer.datamodule)

    # make optimizer private
    pl_module.configure_optimizers_old = pl_module.configure_optimizers
    pl_module.configure_optimizers = MethodType(configure_optimizers, pl_module)
