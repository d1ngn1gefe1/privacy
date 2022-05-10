from torchvision.datasets import CIFAR10, CIFAR100

from .base_datamodule import BaseDataModule
from data.transforms import get_transform


num_classes = {'cifar10': 10, 'cifar100': 100}
CIFAR = {'cifar10': CIFAR10, 'cifar100': CIFAR100}


class CIFARDataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = num_classes[cfg.dataset]
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    CIFAR[self.cfg.dataset](self.cfg.dir_data, train=True, download=True)
    CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, download=True)

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)
    self.dataset_train = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=True, transform=transform_train)
    self.dataset_val = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, transform=transform_val)
    self.dataset_test = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, transform=transform_test)
    self.dataset_predict = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, transform=transform_test)
