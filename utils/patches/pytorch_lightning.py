from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils.uniform_sampler import DistributedUniformWithReplacementSampler
from opacus.utils.batch_memory_manager import BatchSplittingSampler
from pytorch_lightning.accelerators.ipu import IPUAccelerator
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.ddp import log as log_ddp
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy
from pytorch_lightning.strategies.ddp_spawn import log as log_ddp_spawn
from pytorch_lightning.strategies.launchers.spawn import _FakeQueue, _SpawnLauncher, _SpawnOutput
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.trainer.connectors.data_connector import DataConnector
from pytorch_lightning.utilities.data import has_iterable_dataset
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_debug
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


@property
def lightning_module(self):
  return unwrap_lightning_module(self.model.module if isinstance(self.model, DPDDP) else self.model) \
      if self.model is not None else None


def _setup_model_ddp(self, model):
  device_ids = self.determine_ddp_device_ids()
  if hasattr(unwrap_lightning_module(model), 'privacy_engine'):
    log_ddp_spawn.detail(f'Setting up DPDDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DPDDP(model)
  else:
    log_ddp_spawn.detail(f'Setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DDP(module=model, device_ids=device_ids, **self._ddp_kwargs)
  return model


def _setup_model_ddp_spawn(self, model):
  device_ids = self.determine_ddp_device_ids()
  if hasattr(unwrap_lightning_module(model), 'privacy_engine'):
    log_ddp.detail(f'Setting up DPDDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DPDDP(model)
  else:
    log_ddp.detail(f'Setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}')
    model = DDP(module=model, device_ids=device_ids, **self._ddp_kwargs)
  return model


def _requires_distributed_sampler(self, dataloader):
  return (
      self.trainer._accelerator_connector.replace_sampler_ddp
      and self.trainer._accelerator_connector.is_distributed
      and not isinstance(dataloader.sampler, DistributedSampler)
      and not isinstance(dataloader.batch_sampler, DistributedUniformWithReplacementSampler)
      and not isinstance(dataloader.batch_sampler, BatchSplittingSampler)
      and not has_iterable_dataset(dataloader)
      and not isinstance(self.trainer.accelerator, IPUAccelerator)
  )


def _collect_rank_zero_results(self, trainer, results):
  rank_zero_debug('Finalizing the DDP spawn environment.')
  checkpoint_callback = trainer.checkpoint_callback
  best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

  if self._strategy.global_rank != 0:
    return None

  weights_path = None

  extra = _FakeQueue()
  if is_overridden('add_to_queue', trainer.lightning_module):
    trainer.lightning_module.add_to_queue(extra)
  self.add_to_queue(trainer, extra)

  return _SpawnOutput(best_model_path, weights_path, trainer.state, results, extra)


def patch_pytorch_lightning():
  # make lightning compatible with opacus
  ParallelStrategy.lightning_module = lightning_module
  DDPStrategy._setup_model = _setup_model_ddp
  DataConnector._requires_distributed_sampler = _requires_distributed_sampler

  # make lightning compatible with optuna
  DDPSpawnStrategy._setup_model = _setup_model_ddp_spawn
  _SpawnLauncher._collect_rank_zero_results = _collect_rank_zero_results
