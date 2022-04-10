from pytorchvideo.transforms import (
  ApplyTransformToKey,
  Div255,
  Normalize,
  Permute,
  RandomResizedCrop,
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


# Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo_trainer/pytorchvideo_trainer/conf/datamodule/transforms/kinetics_classification_mvit_16x4.yaml
def get_mvit_transforms(T=16):
  transform_train = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T),
        Div255(),
        Permute(dims=(1, 0, 2, 3)),
        RandAugment(magnitude=7, num_layers=4),
        Permute(dims=(1, 0, 2, 3)),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        RandomResizedCrop(target_height=224, target_width=224, scale=(0.08, 1.0), aspect_ratio=(0.75, 1.3333)),
        RandomHorizontalFlip(p=0.5),
        Permute(dims=(1, 0, 2, 3)),
        RandomErasing(probability=0.25, mode='pixel', max_count=1, num_splits=1, device='cpu'),
        Permute(dims=(1, 0, 2, 3)),
      ]
    )
  )

  transform_val = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T),
        Div255(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ShortSideScale(224),
        CenterCrop(224)
      ]
    )
  )

  transform_test = ApplyTransformToKey(
    key='video',
    transform=Compose(
      transforms=[
        UniformTemporalSubsample(num_samples=T),
        Div255(),
        Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ShortSideScale(224),
        CenterCrop(224)
      ]
    )
  )

  return transform_train, transform_val, transform_test
