from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  Permute,
  RandomResizedCrop,
  RemoveKey,
  ShortSideScale,
  UniformTemporalSubsample
)
from pytorchvideo.transforms.rand_augment import RandAugment
from pytorchvideo_trainer.datamodule.rand_erase_transform import RandomErasing
from torchvision.transforms import (
  CenterCrop,
  Compose,
  RandomHorizontalFlip
)

from .constants import *


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml
def get_mvit_transforms(cfg):
  transform_train = Compose(transforms=[
    ApplyTransformToKey(
      key='video',
      transform=Compose(
          transforms=[
            UniformTemporalSubsample(num_samples=cfg.T),
            Div255(),
            Permute(dims=(1, 0, 2, 3)),
            RandAugment(magnitude=7, num_layers=4),
            Permute(dims=(1, 0, 2, 3)),
            Normalize(mean=MEAN_KINETICS, std=STD_KINETICS),
            RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
            RandomHorizontalFlip(p=0.5),
            Permute(dims=(1, 0, 2, 3)),
            RandomErasing(probability=0.25, mode='pixel', max_count=1, num_splits=1, device='cpu'),
            Permute(dims=(1, 0, 2, 3)),
          ]
        )
    ),
    RemoveKey(key='audio')
  ])

  transform_val = Compose(transforms=[
    ApplyTransformToKey(
      key='video',
      transform=Compose(
        transforms=[
          UniformTemporalSubsample(num_samples=cfg.T),
          Div255(),
          Normalize(mean=MEAN_KINETICS, std=STD_KINETICS),
          ShortSideScale(224),
          CenterCrop(224)
        ]
      )
    ),
    RemoveKey(key='audio')]
  )

  transform_test = transform_val
  if not cfg.augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
