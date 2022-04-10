from .datasets import CIFARDataModule, CheXpertDataModule, MedMNISTDataModule, UCF101DataModule
from .dp import make_private


def get_data(cfg):
  if cfg.dataset == 'cifar10' or cfg.dataset == 'cifar100':
    data = CIFARDataModule(cfg)
  elif cfg.dataset == 'chexpert':
    data = CheXpertDataModule(cfg)
  elif cfg.dataset.endswith('mnist'):
    data = MedMNISTDataModule(cfg)
  elif cfg.dataset == 'ucf101':
    data = UCF101DataModule(cfg)
  else:
    raise NotImplementedError

  if cfg.dp:
    data = make_private(data)

  return data
