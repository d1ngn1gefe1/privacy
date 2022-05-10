import medmnist
from medmnist import INFO

from .base_datamodule import BaseDataModule
from data.transforms import get_transform


class MedMNISTDataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    info = INFO[cfg.dataset]
    task = info['task'].split(',')[0]
    num_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])
    assert task in ['multi-class', 'multi-label', 'ordinal-regression'], f'Task {task} not supported'
    if task == 'ordinal-regression':
      task = 'multi-class'

    self.cfg.num_classes = num_classes
    self.cfg.task = task
    self.DataClass = DataClass
    self.target_transform = (lambda x: x[0]) if task == 'multi-class' else None

  def prepare_data(self):
    self.DataClass(root=self.cfg.dir_data, split='train', as_rgb=True, download=True)
    self.DataClass(root=self.cfg.dir_data, split='val', as_rgb=True, download=True)
    self.DataClass(root=self.cfg.dir_data, split='test', as_rgb=True, download=True)

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)
    self.dataset_train = self.DataClass(root=self.cfg.dir_data, split='train', as_rgb=True,
                                        transform=transform_train, target_transform=self.target_transform)
    self.dataset_val = self.DataClass(root=self.cfg.dir_data, split='val', as_rgb=True,
                                      transform=transform_val, target_transform=self.target_transform)
    self.dataset_test = self.DataClass(root=self.cfg.dir_data, split='test', as_rgb=True,
                                       transform=transform_test, target_transform=self.target_transform)
