from .cifar import CIFAR10DataModule
from .lightning import DPLightningDataModule


def get_data(cfg):
  if cfg.dataset == 'cifar10':
    data = CIFAR10DataModule(cfg)
  else:
    raise NotImplementedError

  if cfg.dp:
    data = DPLightningDataModule(data)
  return data

