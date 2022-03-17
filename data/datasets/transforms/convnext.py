from torchvision.transforms import (
  CenterCrop,
  Compose,
  Normalize,
  RandomResizedCrop,
  RandomHorizontalFlip,
  Resize,
  ToTensor
)

from .constants import *


def get_convnext_transforms(augment):
  if augment:
    transform_train = Compose([
      RandomResizedCrop(224, scale=(0.05, 1.0)),
      RandomHorizontalFlip(),
      ToTensor(),
      Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
    ])
  else:
    transform_train = Compose([
      Resize(224),
      CenterCrop(224),
      ToTensor(),
      Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
    ])

  transform_val = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
  ])

  transform_test = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
  ])

  return transform_train, transform_val, transform_test
