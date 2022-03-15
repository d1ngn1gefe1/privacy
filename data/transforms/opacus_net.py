from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor
)

from .constants import *


def get_opacus_net_transforms(augment):
  if augment:
    transform_train = Compose([
      RandomCrop(32, padding=4),
      RandomHorizontalFlip(),
      ToTensor(),
      Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
    ])
  else:
    transform_train = Compose([
      ToTensor(),
      Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
    ])

  transform_val = Compose([
    ToTensor(),
    Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
  ])

  transform_test = Compose([
    ToTensor(),
    Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
  ])

  return transform_train, transform_val, transform_test
