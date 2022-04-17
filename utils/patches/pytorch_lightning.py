from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.uniform_sampler import DistributedUniformWithReplacementSampler
from pytorch_lightning.accelerators.ipu import IPUAccelerator
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.strategies.ddp import DDPStrategy, log
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.utilities.data import has_iterable_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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


def _requires_distributed_sampler(self, dataloader):
  return (
      self.trainer._accelerator_connector.replace_sampler_ddp
      and self.trainer._accelerator_connector.is_distributed
      and not isinstance(dataloader.sampler, DistributedSampler)
      and not isinstance(dataloader.batch_sampler, DistributedUniformWithReplacementSampler)
      and not has_iterable_dataset(dataloader)
      and not isinstance(self.trainer.accelerator, IPUAccelerator)
  )


def patch_pytorch_lightning():
  # make lightning compatible with opacus
  ParallelStrategy.lightning_module = lightning_module
  DDPStrategy._setup_model = _setup_model
  DataConnector._requires_distributed_sampler = _requires_distributed_sampler
