from opacus.data_loader import DPDataLoader
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.data_loading import TrainerDataLoadingMixin
from types import MethodType


# Reference: https://github.com/pytorch/opacus/blob/main/opacus/privacy_engine.py#L153
def train_dataloader(self) -> DPDataLoader:
  dataloader = DPDataLoader.from_data_loader(self.train_dataloader_old(), distributed=len(self.cfg.gpus) > 1)
  # print(f'Dataloader: type={type(dataloader)}, len_dataloader={len(dataloader)}, len_dataset={len(dataloader.dataset)}')
  return dataloader


# Reference: https://github.com/pytorch/opacus/blob/main/examples/mnist_lightning.py
def make_private(data: LightningDataModule) -> LightningDataModule:
  # distributed sampler is handled by Opacus already
  TrainerDataLoadingMixin._requires_distributed_sampler = lambda x, y: False

  # replace dataloader
  data.train_dataloader_old = data.train_dataloader
  data.train_dataloader = MethodType(train_dataloader, data)

  return data
