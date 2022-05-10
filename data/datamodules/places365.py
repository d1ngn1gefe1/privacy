import os
import os.path as osp
from torchvision.datasets import Places365

from .base_datamodule import BaseDataModule
from data.transforms import get_transform


class Places365DataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = 365
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    dir_places365 = osp.join(self.cfg.dir_data, 'places365')
    dir_train = osp.join(dir_places365, 'data_large_standard')
    dir_val = osp.join(dir_places365, 'val_large')

    os.makedirs(dir_places365, exist_ok=True)
    Places365(dir_places365, split='train-standard', download=not osp.exists(dir_train))
    Places365(dir_places365, split='val', download=not osp.exists(dir_val))

  def setup(self, stage=None):
    dir_places365 = osp.join(self.cfg.dir_data, 'places365')
    transform_train, transform_val, transform_test = get_transform(self.cfg)
    self.dataset_train = Places365(dir_places365, split='train-standard', transform=transform_train)
    self.dataset_val = Places365(dir_places365, split='val', transform=transform_val)
    self.dataset_test = Places365(dir_places365, split='val', transform=transform_test)
