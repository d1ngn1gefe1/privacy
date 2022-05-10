import os.path as osp
from torchvision.datasets import ImageFolder

from .base_datamodule import BaseDataModule
from data.transforms import get_transform


class ImageNetDataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = 1000
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    # Tutorial: https://github.com/facebookarchive/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset
    assert osp.isdir(osp.join(self.cfg.dir_data, 'imagenet/train')) and \
           osp.isdir(osp.join(self.cfg.dir_data, 'imagenet/val'))

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)
    self.dataset_train = ImageFolder(osp.join(self.cfg.dir_data, 'imagenet/train'), transform=transform_train)
    self.dataset_val = ImageFolder(osp.join(self.cfg.dir_data, 'imagenet/val'), transform=transform_val)
    self.dataset_test = ImageFolder(osp.join(self.cfg.dir_data, 'imagenet/val'), transform=transform_test)
