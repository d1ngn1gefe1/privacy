from opacus.data_loader import DPDataLoader
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from types import MethodType

from .cifar import CIFAR10DataModule


# Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L153
def train_dataloader(self):
  return DPDataLoader.from_data_loader(self.train_dataloader_old(), distributed=len(self.cfg.gpus) > 1)


def get_data(cfg):
  if cfg.dataset == 'cifar10':
    data = CIFAR10DataModule(cfg)
  else:
    raise NotImplementedError

  if cfg.dp:
    # distributed sampler is handled by Opacus already
    TrainerDataLoadingMixin._requires_distributed_sampler = lambda x, y: False

    # replace dataloader
    data.train_dataloader_old = data.train_dataloader
    data.train_dataloader = MethodType(train_dataloader, data)

  return data

