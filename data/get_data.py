from .cifar import CIFAR10DataModule
from .dp import make_private


def get_data(cfg):
  if cfg.dataset == 'cifar10':
    data = CIFAR10DataModule(cfg)
  else:
    raise NotImplementedError

  if cfg.dp:
    data = make_private(data)

  return data

