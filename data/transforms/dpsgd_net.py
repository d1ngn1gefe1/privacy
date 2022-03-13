from torchvision.transforms import (
  CenterCrop,
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor
)


def get_dpsgd_net_transforms():
  transform_train = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    RandomCrop(24),
    RandomHorizontalFlip()
  ])

  transform_val = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    CenterCrop(24),
  ])

  transform_test = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    CenterCrop(24),
  ])

  return transform_train, transform_val, transform_test
