from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from .transforms import get_transforms


num_classes = {'cifar10': 10, 'cifar100': 100}
CIFAR = {'cifar10': CIFAR10, 'cifar100': CIFAR100}


class CIFARDataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()
    cfg.num_classes = num_classes[cfg.dataset]
    self.cfg = cfg

  def prepare_data(self):
    CIFAR[self.cfg.dataset](self.cfg.dir_data, train=True, download=True)
    CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, download=True)

  def setup(self, stage=None):
    transforms_train, transforms_val, transforms_test = get_transforms(self.cfg.net, self.cfg.augment)
    self.dataset_train = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=True, transform=transforms_train)
    self.dataset_val = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, transform=transforms_val)
    self.dataset_test = CIFAR[self.cfg.dataset](self.cfg.dir_data, train=False, transform=transforms_test)

  def train_dataloader(self):
    num_gpus = len(self.cfg.gpus)
    dataloader = DataLoader(self.dataset_train, batch_size=self.cfg.batch_size//num_gpus,
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    print(f'len_dataloader={len(dataloader)}, len_dataset={len(self.dataset_train)}')
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
