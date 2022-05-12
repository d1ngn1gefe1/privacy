from torchvision.transforms import (
  CenterCrop,
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor
)

from .constants import *
from .transforms import ApplyTransformOnList, Repeat


def get_dpsgd_net_transforms(cfg):
  transform_train = Compose([
    RandomCrop(24),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=MEAN_DEFAULT, std=STD_DEFAULT)
  ])

  if hasattr(cfg, 'num_views'):
    transform_train = Compose([
      Repeat(cfg.num_views),
      ApplyTransformOnList(transform=transform_train)
    ])

  transform_val = Compose([
    CenterCrop(24),
    ToTensor(),
    Normalize(mean=MEAN_DEFAULT, std=STD_DEFAULT)
  ])

  transform_test = transform_val
  if not cfg.augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
