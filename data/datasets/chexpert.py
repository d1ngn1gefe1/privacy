import numpy as np
import os
import pandas as pd
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple

from .transforms import get_transforms


class CheXpert(VisionDataset):
  def __init__(self, root: str, train: bool = True,
               transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
    super().__init__(root, transform=transform, target_transform=target_transform)

    data = pd.read_csv(os.path.join(root, 'CheXpert-v1.0-small', 'train.csv' if train else 'valid.csv'))

    paths = [os.path.join(root, x) for x in data.Path.values]
    assert all(os.path.exists(path) for path in paths), 'Dataset not found or corrupted'

    targets = data.iloc[:, 5:].values
    targets = np.nan_to_num(targets).astype(int)
    targets[targets == -1] = 0
    assert len(paths) == len(targets), 'Dataset not found or corrupted'

    self.paths = paths
    self.targets = targets
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    image = Image.open(self.paths[index]).convert('RGB')
    target = self.targets[index]

    if self.transform is not None:
      image = self.transform(image)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return image, target


class CheXpertDataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    cfg.num_classes = 14
    self.cfg = cfg

  def prepare_data(self):
    print('Please download the dataset on https://stanfordmlgroup.github.io/competitions/chexpert/')

  def setup(self, stage=None):
    transforms_train, transforms_val, transforms_test = get_transforms(self.cfg.net, self.cfg.augment)
    self.dataset_train = CheXpert(self.cfg.dir_data, train=True, transform=transforms_train)
    self.dataset_val = CheXpert(self.cfg.dir_data, train=False, transform=transforms_val)
    self.dataset_test = CheXpert(self.cfg.dir_data, train=False, transform=transforms_test)

  def train_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    dataloader = DataLoader(self.dataset_train, batch_size=self.cfg.batch_size//num_gpus,
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return dataloader

  def val_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    dataloader = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size//num_gpus,
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return dataloader

  def test_loader(self):
    num_gpus = len(self.cfg.gpus)
    dataloader = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size//num_gpus,
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return dataloader
