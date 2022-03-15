from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from .transforms import get_transforms


class CIFAR10DataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    cfg.num_classes = 10
    self.cfg = cfg

  def prepare_data(self):
    CIFAR10(self.cfg.dir_data, train=True, download=True)
    CIFAR10(self.cfg.dir_data, train=False, download=True)

  def setup(self, stage=None):
    transforms_train, transforms_val, transforms_test = get_transforms(self.cfg.net, self.cfg.augment)
    self.cifar10_train = CIFAR10(self.cfg.dir_data, train=True, transform=transforms_train)
    self.cifar10_val = CIFAR10(self.cfg.dir_data, train=False, transform=transforms_val)
    self.cifar10_test = CIFAR10(self.cfg.dir_data, train=False, transform=transforms_test)

  def train_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    cifar10_train = DataLoader(self.cifar10_train, batch_size=self.cfg.batch_size//num_gpus,
                               num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return cifar10_train

  def val_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    cifar10_val = DataLoader(self.cifar10_val, batch_size=self.cfg.batch_size//num_gpus,
                             num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return cifar10_val

  def test_loader(self):
    num_gpus = len(self.cfg.gpus)
    cifar10_test = DataLoader(self.cifar10_val, batch_size=self.cfg.batch_size//num_gpus,
                              num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return cifar10_test


class CIFAR100DataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    cfg.num_classes = 100
    self.cfg = cfg

  def prepare_data(self):
    CIFAR100(self.cfg.dir_data, train=True, download=True)
    CIFAR100(self.cfg.dir_data, train=False, download=True)

  def setup(self, stage=None):
    transforms_train, transforms_val, transforms_test = get_transforms(self.cfg.net, self.cfg.augment)
    self.cifar100_train = CIFAR100(self.cfg.dir_data, train=True, transform=transforms_train)
    self.cifar100_val = CIFAR100(self.cfg.dir_data, train=False, transform=transforms_val)
    self.cifar100_test = CIFAR100(self.cfg.dir_data, train=False, transform=transforms_test)

  def train_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    cifar100_train = DataLoader(self.cifar100_train, batch_size=self.cfg.batch_size//num_gpus,
                                num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return cifar100_train

  def val_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    cifar100_val = DataLoader(self.cifar100_val, batch_size=self.cfg.batch_size//num_gpus,
                              num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return cifar100_val

  def test_loader(self):
    num_gpus = len(self.cfg.gpus)
    cifar100_test = DataLoader(self.cifar100_val, batch_size=self.cfg.batch_size//num_gpus,
                               num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return cifar100_test
