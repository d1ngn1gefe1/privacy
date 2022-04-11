from opacus.data_loader import DPDataLoader
from pytorch_lightning import LightningDataModule
from types import MethodType


# Reference: opacus -> privacy_engine -> make_private -> _prepare_data_loader
def train_dataloader(self) -> DPDataLoader:
  dataloader = DPDataLoader.from_data_loader(self.train_dataloader_old(), distributed=len(self.cfg.gpus) > 1)
  print(f'Dataloader: type={type(dataloader)}')
  return dataloader


# Reference: opacus -> lightning
def make_private(data: LightningDataModule) -> LightningDataModule:
  # make train dataloader private
  data.train_dataloader_old = data.train_dataloader
  data.train_dataloader = MethodType(train_dataloader, data)

  return data
