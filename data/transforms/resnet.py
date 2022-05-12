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
from .transforms import ApplyTransformOnList, Repeat


def get_resnet_transforms(cfg):
  transform_train = Compose([
    RandomResizedCrop(224, scale=(0.05, 1.0)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
  ])

  if hasattr(cfg, 'num_views'):
    transform_train = Compose([
      Repeat(cfg.num_views),
      ApplyTransformOnList(transform=transform_train)
    ])

  transform_val = Compose([
    Resize(224),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
  ])

  transform_test = transform_val
  if not cfg.augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
