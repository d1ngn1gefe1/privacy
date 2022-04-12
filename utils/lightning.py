from opacus.grad_sample import GradSampleModule
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.strategies.ddp import DDPStrategy, log
from pytorch_lightning.strategies.parallel import ParallelStrategy
from torch.nn.parallel import DistributedDataParallel as DDP


@property
def lightning_module(self):
  if not self.model:
    return None

  if isinstance(self.model, DPDDP):
    return unwrap_lightning_module(self.model.module)
  elif isinstance(self.model, GradSampleModule):
    return unwrap_lightning_module(self.model._module.module)
  else:
    return self.model
  #return unwrap_lightning_module(self.model.module if isinstance(self.model, DPDDP) else self.model) \
  #    if self.model is not None else None


def _setup_model(self, model):
  device_ids = self.determine_ddp_device_ids()
  if model.module.cfg.dp:
    log.detail(f'setting up DPDDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DPDDP(model)
  else:
    log.detail(f'setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DDP(module=model, device_ids=device_ids, **self._ddp_kwargs)

  return model


def patch_lightning():
  # make lightning compatible
  ParallelStrategy.lightning_module = lightning_module
  DDPStrategy._setup_model = _setup_model
