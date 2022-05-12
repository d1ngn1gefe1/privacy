from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor
)

from .constants import *
from .transforms import ApplyTransformOnList, Repeat


def get_opacus_net_transforms(cfg):
  transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
  ])

  if hasattr(cfg, 'num_views'):
    transform_train = Compose([
      Repeat(cfg.num_views),
      ApplyTransformOnList(transform=transform_train)
    ])

  transform_val = Compose([
    ToTensor(),
    Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
  ])

  transform_test = transform_val
  if not cfg.augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
