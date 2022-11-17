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
  elif cfg.dataset == 'places365':
    from .places365 import Places365DataModule
    data = Places365DataModule(cfg)
  elif cfg.dataset == 'ucf101':
    from .ucf101 import UCF101DataModule
    data = UCF101DataModule(cfg)
  elif cfg.dataset == 'ssv2' or cfg.dataset == 'ssv2mini':
    from .ssv2 import SSv2DataModule
    data = SSv2DataModule(cfg)
  elif cfg.dataset == 'diving48':
    from .diving48 import Diving48DataModule
    data = Diving48DataModule(cfg)
  elif cfg.dataset == 'hmdb51':
    from .hmdb51 import HMDB51DataModule
    data = HMDB51DataModule(cfg)
  else:
    raise NotImplementedError

  return data
