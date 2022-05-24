import clip
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


def get_vit_transforms(cfg):
  if cfg.weight == 'pretrain_clip':
    _, transform_train = clip.load('ViT-B/16')
    transform_val = transform_train
    transform_test = transform_train
  else:
    transform_train = Compose([
      RandomResizedCrop(224, scale=(0.05, 1.0)),
      RandomHorizontalFlip(),
      ToTensor(),
      Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
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
