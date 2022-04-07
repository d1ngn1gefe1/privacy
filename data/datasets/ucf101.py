import csv
import os
from pytorchvideo.data import Ucf101, make_clip_sampler
from pytorch_lightning import LightningDataModule
import rarfile
import ssl
import torch
from torch.utils.data import DistributedSampler, RandomSampler, DataLoader
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

  url_metadata = 'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
  md5_metadata = '4426308e815a5108ba976f4a2bb47c5e'
  dname_metadata = 'ucfTrainTestlist'

  def __init__(self, cfg):
    super().__init__()
    cfg.num_classes = 101
    cfg.task = 'multi-class'
    self.cfg = cfg

  def prepare_data(self):
    # download and extract
    ssl._create_default_https_context = ssl._create_unverified_context
    if not os.path.isdir(os.path.join(self.cfg.dir_data, self.dname_video)):
      download_and_extract_archive(self.url_video, self.cfg.dir_data, md5=self.md5_video)
    if not os.path.isdir(os.path.join(self.cfg.dir_data, self.dname_metadata)):
      download_and_extract_archive(self.url_metadata, self.cfg.dir_data, md5=self.md5_metadata)

    if os.path.exists(os.path.join(self.cfg.dir_data, self.dname_metadata, 'trainlist01.csv')) and \
       os.path.exists(os.path.join(self.cfg.dir_data, self.dname_metadata, 'testlist01.csv')):
      return

    # generate labeled video paths
    with open(os.path.join(self.cfg.dir_data, self.dname_metadata, 'classInd.txt')) as f:
      data = f.readlines()
      data = [x.strip().split(' ') for x in data]
      cname_to_cid = {x[1]:int(x[0])-1 for x in data}

    for split in ['train', 'test']:
      with open(os.path.join(self.cfg.dir_data, self.dname_metadata, f'{split}list01.txt')) as f:
        data = f.readlines()
        cnames = [x.strip().split('/')[0] for x in data]
        cids = [cname_to_cid[x] for x in cnames]
        rows = [[x.strip().split(' ')[0], y] for x, y in zip(data, cids)]

      with open(os.path.join(self.cfg.dir_data,  self.dname_metadata, f'{split}list01.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(rows)

  def setup(self, stage=None):
    # pytorch-lightning does not handle iterable datasets
    # Reference: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#replace-sampler-ddp
    if torch.distributed.is_available() and torch.distributed.is_initialized():
      video_sampler = DistributedSampler
    else:
      video_sampler = RandomSampler

    transform_train, transform_val, transform_test = get_transforms('mvit', True)

    self.dataset_train = Ucf101(
      data_path=os.path.join(self.cfg.dir_data, self.dname_metadata, 'trainlist01.csv'),
      clip_sampler=make_clip_sampler('random', self.cfg.T*self.cfg.tau/self.cfg.fps),
      video_sampler=video_sampler,
      transform=transform_train,
      video_path_prefix=os.path.join(self.cfg.dir_data, self.dname_video),
      decode_audio=False
    )

    self.dataset_val = Ucf101(
      data_path=os.path.join(self.cfg.dir_data, self.dname_metadata, 'testlist01.csv'),
      clip_sampler=make_clip_sampler('constant_clips_per_video', self.cfg.T*self.cfg.tau/self.cfg.fps, 3, 1),
      video_sampler=video_sampler,
      transform=transform_val,
      video_path_prefix=os.path.join(self.cfg.dir_data, self.dname_video),
      decode_audio=False
    )

    self.dataset_test = Ucf101(
      data_path=os.path.join(self.cfg.dir_data, self.dname_metadata, 'testlist01.csv'),
      clip_sampler=make_clip_sampler('constant_clips_per_video', self.cfg.T*self.cfg.tau/self.cfg.fps, 3, 1),
      video_sampler=video_sampler,
      transform=transform_test,
      video_path_prefix=os.path.join(self.cfg.dir_data, self.dname_video),
      decode_audio=False
    )

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

