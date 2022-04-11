import inspect
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.privacy_engine import forbid_accumulation_hook
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.strategies.ddp import DDPStrategy, log
from pytorch_lightning.strategies.parallel import ParallelStrategy
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from types import MethodType


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(delta=self.cfg.delta)
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=False, prog_bar=True)  # same across all devices, no sync


def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
  sample_rate = 1/len(dataloader)
  if len(self.cfg.gpus) > 1:
    distributed = True
    batch_size = int(len(dataloader.dataset)*sample_rate)/torch.distributed.get_world_size()
  else:
    distributed = False
    batch_size = int(len(dataloader.dataset)*sample_rate)

  dict_optimizers = self.configure_optimizers_old()
  optimizer_old, scheduler_old = dict_optimizers['optimizer'], dict_optimizers['lr_scheduler']

  # optimizer
  optimizer = self.privacy_engine._prepare_optimizer(optimizer_old,
                                                     distributed=distributed,
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=self.cfg.c,
                                                     expected_batch_size=batch_size,
                                                     clipping='flat')
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

  # lr scheduler
  kwargs = {key:scheduler_old.__dict__[key]
            for key in inspect.signature(scheduler_old.__class__.__init__).parameters.keys()
            if key not in ['self', 'optimizer']}
  scheduler = scheduler_old.__class__(optimizer, **kwargs)

  print(f'Net: {type(self.net)}, Optimizer: {type(optimizer)}')
  return {'optimizer': optimizer, 'lr_scheduler': scheduler}


# TODO 1
@property
def lightning_module(self):
  return unwrap_lightning_module(self.model) if self.model is not None else None


def tmp(model):
  privacy_engine = PrivacyEngine()

  # make optimizer private
  model.configure_optimizers_old = model.configure_optimizers
  model.configure_optimizers = MethodType(configure_optimizers, model)

  # make net private
  model.net = privacy_engine._prepare_model(model.net)
  model.net.get_classifier = model.net._module.get_classifier
  # model.net.register_forward_pre_hook(forbid_accumulation_hook)  # TODO 3

  # attach privacy engine
  model.privacy_engine = PrivacyEngine()
  model.on_train_epoch_end = MethodType(on_train_epoch_end, model)


def _setup_model(self, model):
  device_ids = self.determine_ddp_device_ids()
  if model.module.cfg.dp:
    log.detail(f'setting up DPDDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DPDDP(model)
  else:
    log.detail(f'setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DDP(module=model, device_ids=device_ids, **self._ddp_kwargs)

  tmp(model.module.module)  # TODO 2

  return model


def make_private(model):
  # make lightning compatible
  ParallelStrategy.lightning_module = lightning_module  # TODO 1
  DDPStrategy._setup_model = _setup_model

  # tmp(model)  # TODO 2

  return model
