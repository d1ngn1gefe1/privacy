from torchvision.transforms import (
  CenterCrop,
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor
)

from .constants import *


def get_dpsgd_net_transforms(augment):
  transform_train = Compose([
    RandomCrop(24),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=MEAN_DEFAULT, std=STD_DEFAULT)
  ])

  transform_val = Compose([
    CenterCrop(24),
    ToTensor(),
    Normalize(mean=MEAN_DEFAULT, std=STD_DEFAULT)
  ])

  transform_test = Compose([
    CenterCrop(24),
    ToTensor(),
    Normalize(mean=MEAN_DEFAULT, std=STD_DEFAULT)
  ])

  if not augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
