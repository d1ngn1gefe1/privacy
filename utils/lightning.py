from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.strategies.ddp import DDPStrategy, log
from pytorch_lightning.strategies.parallel import ParallelStrategy
from torch.nn.parallel import DistributedDataParallel as DDP


@property
def lightning_module(self):
  return unwrap_lightning_module(self.model.module if isinstance(self.model, DPDDP) else self.model) \
      if self.model is not None else None


def _setup_model(self, model):
  device_ids = self.determine_ddp_device_ids()
  if hasattr(unwrap_lightning_module(model), 'privacy_engine'):
    log.detail(f'setting up DPDDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DPDDP(model)
  else:
    log.detail(f'setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DDP(module=model, device_ids=device_ids, **self._ddp_kwargs)
  return model


# make lightning compatible with opacus
def patch_lightning():
  ParallelStrategy.lightning_module = lightning_module
  DDPStrategy._setup_model = _setup_model
