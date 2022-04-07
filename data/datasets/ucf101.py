import os
from pytorch_lightning import LightningDataModule
import rarfile
import ssl
from torch.utils.data import DataLoader
import torchvision.datasets.utils
from torchvision.datasets.utils import download_and_extract_archive, _ARCHIVE_EXTRACTORS

from .transforms import get_transforms


def _extract_rar(from_path, to_path, compression):
  with rarfile.RarFile(from_path) as f:
    f.extractall(to_path)


_ARCHIVE_EXTRACTORS['.rar'] = _extract_rar
torchvision.datasets.utils._ARCHIVE_EXTRACTORS = _ARCHIVE_EXTRACTORS


class UCF101DataModule(LightningDataModule):
  url_video = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
  md5_video = '6463accf4bbc20fa6418ab7f159e6149'
  dname_video = 'UCF-101'

  url_split = 'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
  md5_split = '4426308e815a5108ba976f4a2bb47c5e'
  dname_split = 'ucfTrainTestlist'

  def __init__(self, cfg):
    super().__init__()
    cfg.num_classes = 101
    cfg.task = 'multi-class'
    self.cfg = cfg

  def prepare_data(self):
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.isdir(os.path.join(self.cfg.dir_data, self.dname_video)):
      download_and_extract_archive(self.url_video, self.cfg.dir_data, md5=self.md5_video)
    if not os.path.isdir(os.path.join(self.cfg.dir_data, self.dname_split)):
      download_and_extract_archive(self.url_split, self.cfg.dir_data, md5=self.md5_split)

  def setup(self, stage=None):
    pass

  def train_dataloader(self):
    pass

  def val_dataloader(self):
    pass

  def test_dataloader(self):
    pass

  def predict_dataloader(self):
    pass

