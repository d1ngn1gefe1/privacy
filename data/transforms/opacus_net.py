from torchvision.transforms import (
  Compose,
  Normalize,
  RandomCrop,
  RandomHorizontalFlip,
  ToTensor
)


def get_opacus_net_transforms(augment):
  if augment:
    transform_train = Compose([
      ToTensor(),
      Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
      RandomCrop(32, padding=4),
      RandomHorizontalFlip()
    ])
  else:
    transform_train = Compose([
      RandomCrop(32, padding=4),
      RandomHorizontalFlip()
    ])

  transform_val = Compose([
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
  ])

  transform_test = Compose([
    ToTensor(),
    Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
  ])

  return transform_train, transform_val, transform_test
