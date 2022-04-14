from abc import ABC
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule, ABC):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.dataset_train, self.dataset_val, self.dataset_test, self.dataset_predict = None, None, None, None

  def prepare_data(self):
    raise NotImplementedError

  def setup(self, stage=None):
    raise NotImplementedError

  def train_dataloader(self):
    dataloader = DataLoader(self.dataset_train, batch_size=self.cfg.batch_size//len(self.cfg.gpus),
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=True)
    return dataloader

  def val_dataloader(self):
    dataloader = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size//len(self.cfg.gpus),
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return dataloader

  def test_dataloader(self):
    dataloader = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size//len(self.cfg.gpus),
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return dataloader

  def predict_dataloader(self):
    dataloader = DataLoader(self.dataset_val, batch_size=self.cfg.batch_size//len(self.cfg.gpus),
                            num_workers=self.cfg.num_workers, pin_memory=True, drop_last=False)
    return dataloader
