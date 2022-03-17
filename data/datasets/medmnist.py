import medmnist
from medmnist import INFO
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .transforms import get_transforms


class MedMNISTDataModule(LightningDataModule):
  def __init__(self, cfg):
    super().__init__()

    info = INFO[cfg.dataset]
    task = info['task']
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    assert task in ['multi-class', 'multi-label', 'ordinal-regression'], f'Task {task} not supported'
    if task == 'ordinal-regression':
      task = 'multi-class'

    cfg.num_classes = num_classes
    cfg.task = task
    self.cfg = cfg
    self.DataClass = DataClass

  def prepare_data(self):
    self.DataClass(root=self.cfg.dir_data, split='train', as_rgb=True, download=True)
    self.DataClass(root=self.cfg.dir_data, split='val', as_rgb=True, download=True)
    self.DataClass(root=self.cfg.dir_data, split='test', as_rgb=True, download=True)

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transforms(self.cfg.net, self.cfg.augment)
    self.dataset_train = self.DataClass(root=self.cfg.dir_data, split='train', as_rgb=True,
                                        transform=transform_train, target_transform=lambda x: x[0])
    self.dataset_val = self.DataClass(root=self.cfg.dir_data, split='val', as_rgb=True,
                                      transform=transform_val, target_transform=lambda x: x[0])
    self.dataset_test = self.DataClass(root=self.cfg.dir_data, split='test', as_rgb=True,
                                       transform=transform_test, target_transform=lambda x: x[0])

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
