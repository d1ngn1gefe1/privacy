import os
from torchvision.datasets import ImageFolder

from .base_datamodule import BaseDataModule
from .transforms import get_transforms


class ImageNetDataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = 1000
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    # Tutorial: https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
    assert os.path.isdir(os.path.join(self.cfg.dir_data, 'imagenet/train')) and \
           os.path.isdir(os.path.join(self.cfg.dir_data, 'imagenet/val'))

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transforms(self.cfg.net, self.cfg.augment)
    self.dataset_train = ImageFolder(os.path.join(self.cfg.dir_data, 'imagenet/train'), transform=transform_train)
    self.dataset_val = ImageFolder(os.path.join(self.cfg.dir_data, 'imagenet/val'), transform=transform_val)
    self.dataset_test = ImageFolder(os.path.join(self.cfg.dir_data, 'imagenet/val'), transform=transform_test)
