from .cifar import CIFAR10DataModule, CIFAR100DataModule
from .dp import make_private


def get_data(cfg):
  if cfg.dataset == 'cifar10':
    data = CIFAR10DataModule(cfg)
  elif cfg.dataset == 'cifar100':
    data = CIFAR100DataModule(cfg)
  else:
    raise NotImplementedError

  if cfg.dp:
    data = make_private(data)

  return data

