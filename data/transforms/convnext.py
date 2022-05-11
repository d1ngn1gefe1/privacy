# Reference: https://github.com/facebookresearch/ConvNeXt/blob/main/datasets.py

from timm.data import create_transform
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

from .constants import *


def get_convnext_transforms(cfg):
  transform_train = create_transform(
    input_size=224,
    is_training=True,
    color_jitter=0.4,
    auto_augment='rand-m9-mstd0.5-inc1',
    interpolation='bicubic',
    re_prob=0.25,
    re_mode='pixel',
    re_count=1,
    mean=MEAN_IMAGENET,
    std=STD_IMAGENET,
  )

  transform_val = Compose([
    Resize(256, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
  ])

  transform_test = transform_val
  if not cfg.augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
