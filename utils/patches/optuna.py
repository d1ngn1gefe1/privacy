from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import Trainer


def on_init_start(self, trainer: Trainer) -> None:
  self.is_ddp_backend = trainer._accelerator_connector.is_distributed
  if self.is_ddp_backend:
    if version.parse(pl.__version__) < version.parse('1.5.0'):
      raise ValueError('PyTorch Lightning>=1.5.0 is required in DDP.')
    if not (
        isinstance(self._trial.study._storage, _CachedStorage)
        and isinstance(self._trial.study._storage._backend, RDBStorage)
    ):
      breakpoint()
      raise ValueError(
        'optuna.integration.PyTorchLightningPruningCallback'
        ' supports only optuna.storages.RDBStorage in DDP.'
      )


def patch_optuna():
  # make Optuna support pytorch-lightning==1.7.0
  PyTorchLightningPruningCallback.on_init_start = on_init_start
