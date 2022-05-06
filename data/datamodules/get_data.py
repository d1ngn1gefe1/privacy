def get_data(cfg):
  if cfg.dataset == 'cifar10' or cfg.dataset == 'cifar100':
    from .cifar import CIFARDataModule
    data = CIFARDataModule(cfg)
  elif cfg.dataset == 'chexpert':
    from .chexpert import CheXpertDataModule
    data = CheXpertDataModule(cfg)
  elif cfg.dataset.endswith('mnist'):
    from .medmnist import MedMNISTDataModule
    data = MedMNISTDataModule(cfg)
  elif cfg.dataset == 'imagenet':
    from .imagenet import ImageNetDataModule
    data = ImageNetDataModule(cfg)
  elif cfg.dataset == 'ucf101':
    from .ucf101 import UCF101DataModule
    data = UCF101DataModule(cfg)
  else:
    raise NotImplementedError

  return data
