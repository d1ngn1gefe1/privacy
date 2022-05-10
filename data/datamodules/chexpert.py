import numpy as np
import os.path as osp
import pandas as pd
from PIL import Image
from torchvision.datasets import VisionDataset
from typing import Any, Tuple

from .base_datamodule import BaseDataModule
from data.transforms import get_transform


class CheXpert(VisionDataset):
  observations = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

  def __init__(self, root, train=True, transform=None, target_transform=None) -> None:
    super().__init__(root, transform=transform, target_transform=target_transform)

    data = pd.read_csv(osp.join(root, 'CheXpert-v1.0-small', 'train.csv' if train else 'valid.csv'))

    paths = [osp.join(root, x) for x in data.Path.values]
    assert all(osp.exists(path) for path in paths), 'Dataset not found or corrupted'

    targets = data[self.observations].values
    targets[np.isnan(targets)] = -1  # NaN and -1 are both stated as unknowns in the dataset
    assert len(paths) == len(targets), 'Dataset not found or corrupted'

    self.paths = paths
    self.targets = targets
    self.transform = transform
    self.target_transform = target_transform

  @classmethod
  def exists(cls, root):
    return osp.isdir(osp.join(root, 'CheXpert-v1.0-small'))

  def __len__(self) -> int:
    return len(self.paths)

  def __getitem__(self, index: int) -> Tuple[Any, Any]:
    image = Image.open(self.paths[index]).convert('RGB')
    target = self.targets[index].astype(np.int32)

    if self.transform is not None:
      image = self.transform(image)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return image, target


class CheXpertDataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = len(CheXpert.observations)
    self.cfg.task = 'multi-label'

  def prepare_data(self):
    if not CheXpert.exists(self.cfg.dir_data):
      raise RuntimeError('Please download the dataset on https://stanfordmlgroup.github.io/competitions/chexpert/.')

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)
    self.dataset_train = CheXpert(self.cfg.dir_data, train=True, transform=transform_train)
    self.dataset_val = CheXpert(self.cfg.dir_data, train=False, transform=transform_val)
    self.dataset_test = CheXpert(self.cfg.dir_data, train=False, transform=transform_test)
