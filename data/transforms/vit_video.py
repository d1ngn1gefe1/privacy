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
from torchvision.transforms import (
  CenterCrop,
  Compose,
  RandomHorizontalFlip
)

from .constants import *
from .transforms import VideoToImage


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml
def get_vit_transforms(cfg):
  transform_train = Compose(transforms=[
    ApplyTransformToKey(
      key='video',
      transform=Compose(
          transforms=[
            UniformTemporalSubsample(num_samples=cfg.T),
            Div255(),
            Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
            RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
            RandomHorizontalFlip(p=0.5),
            VideoToImage()
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
          Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET),
          ShortSideScale(224),
          CenterCrop(224),
          VideoToImage()
        ]
      )
    ),
    RemoveKey(key='audio')]
  )

  transform_test = transform_val
  if not cfg.augment:
    transform_train = transform_val

  return transform_train, transform_val, transform_test
