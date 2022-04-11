import inspect
import math
from opacus import PrivacyEngine
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import pytorch_lightning as pl
from pytorch_lightning.overrides.base import unwrap_lightning_module, _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.parallel import ParallelStrategy
import torch
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP
from types import MethodType


def on_train_epoch_end(self):
  epsilon = self.privacy_engine.get_epsilon(delta=self.cfg.delta)
  self.log('epsilon', epsilon, on_epoch=True, sync_dist=False, prog_bar=True)  # same across all devices, no sync


def unwrap_lightning_module(wrapped_model):
  """Recursively unwraps a :class:`~pytorch_lightning.core.lightning.LightningModule` by following the
  ``.module`` attributes on the wrapper.
  Raises:
    TypeError: If the unwrapping leads to a module that is not a LightningModule and that cannot be unwrapped
      further.
  """
  model = wrapped_model
  if isinstance(model, (DPDDP, DDP, DP)):
    model = unwrap_lightning_module(model.module)
  if isinstance(model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)):
    model = unwrap_lightning_module(model.module)
  if not isinstance(model, pl.LightningModule):
    raise TypeError(f"Unwrapping the module did not yield a `LightningModule`, got {type(model)} instead.")
  return model


@property
def lightning_module(self):
  return unwrap_lightning_module(self.model) if self.model is not None else None
ParallelStrategy.lightning_module = lightning_module


def configure_optimizers(self):
  dataloader = self.trainer._data_connector._train_dataloader_source.dataloader()
  sample_rate = 1/len(dataloader)
  expected_batch_size = int(len(dataloader.dataset)*sample_rate)
  # TODO: verify ddp batch size
  # print(f'{sample_rate}, {len(dataloader)}, {len(dataloader.dataset)}')

  if len(self.cfg.gpus) > 1:
    distributed = True
    world_size = torch.distributed.get_world_size()
    expected_batch_size /= world_size  # expected_batch_size is the per-worker batch size
    num_layers = len([(n, p) for n, p in self.named_parameters() if p.requires_grad])
    max_grad_norm = [self.cfg.c/math.sqrt(num_layers)]*num_layers
  else:
    distributed = False
    max_grad_norm = self.cfg.c
  clipping = 'flat'

  dict_optimizers = self.configure_optimizers_old()
  optimizer_old, scheduler_old = dict_optimizers['optimizer'], dict_optimizers['lr_scheduler']

  # optimizer
  optimizer = self.privacy_engine._prepare_optimizer(optimizer_old,
                                                     distributed=distributed,
                                                     noise_multiplier=self.cfg.sigma,
                                                     max_grad_norm=max_grad_norm,
                                                     expected_batch_size=expected_batch_size,
                                                     clipping=clipping)
  optimizer.attach_step_hook(self.privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=sample_rate))

  # lr scheduler
  kwargs = {key:scheduler_old.__dict__[key]
            for key in inspect.signature(scheduler_old.__class__.__init__).parameters.keys()
            if key not in ['self', 'optimizer']}
  scheduler = scheduler_old.__class__(optimizer, **kwargs)

  print(f'Net: {type(self.net)}, Optimizer: {type(optimizer)}')
  return {'optimizer': optimizer, 'lr_scheduler': scheduler}


# Reference: https://github.com/pytorch/opacus/blob/main/examples/mnist_lightning.py
def make_private(model, cfg):
  if len(cfg.gpus) > 1:
    # LightningDistributed -> original model
    plain_model = model.module
    model = DPDDP(model)
  else:
    plain_model = model

  plain_model.privacy_engine = PrivacyEngine()
  #model.privacy_engine = plain_model.privacy_engine  # TODO
  plain_model.on_train_epoch_end = MethodType(on_train_epoch_end, plain_model)

  # Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L153
  plain_model.net = plain_model.privacy_engine._prepare_model(plain_model.net)
  plain_model.net.get_classifier = plain_model.net._module.get_classifier
  # plain_model.net.register_forward_pre_hook(forbid_accumulation_hook)  # todo: lightning will trigger an error

  # Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L120
  plain_model.configure_optimizers_old = plain_model.configure_optimizers
  plain_model.configure_optimizers = MethodType(configure_optimizers, plain_model)

  return model


def _setup_model(self, model):
  device_ids = self.determine_ddp_device_ids()
  if model.module.cfg.dp:
    # DPDDP
    #log.detail(f"setting up DPDDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
    return make_private(model, model.module.cfg)
  else:
    #log.detail(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
    return DDP(module=model, device_ids=device_ids, **self._ddp_kwargs)


DDPStrategy._setup_model = _setup_model

