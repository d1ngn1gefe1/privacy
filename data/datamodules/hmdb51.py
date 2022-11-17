import csv
import glob
import os
import os.path as osp
import pandas as pd
from pytorchvideo.data import labeled_video_dataset, make_clip_sampler
import rarfile
from torch.utils.data import DistributedSampler, RandomSampler
import torchvision.datasets.utils
from torchvision.datasets.utils import extract_archive, _ARCHIVE_EXTRACTORS
import wget

from .base_datamodule import BaseDataModule
from data.transforms import get_transform
from .map_dataset import MapDataset
import utils


def _extract_rar(from_path, to_path, compression):
  with rarfile.RarFile(from_path) as f:
    f.extractall(to_path)

  for path in glob.glob(osp.join(to_path, '*.rar')):
    with rarfile.RarFile(path) as f:
      f.extractall(to_path)
      os.remove(path)


_ARCHIVE_EXTRACTORS['.rar'] = _extract_rar
torchvision.datasets.utils._ARCHIVE_EXTRACTORS = _ARCHIVE_EXTRACTORS


class HMDB51DataModule(BaseDataModule):
  url_video = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'
  url_split = 'http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar'

  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = 51
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    dir_hmdb51 = osp.join(self.cfg.dir_data, 'HMDB-51')

    # download and extract videos
    path_rar = osp.join(dir_hmdb51, 'hmdb51_org.rar')
    dir_video = osp.join(dir_hmdb51, 'videos')
    if not osp.exists(path_rar):
      print('Downloading videos.')
      wget.download(self.url_video, path_rar)  # may need to download in terminal
    if not osp.isdir(dir_video):
      print('Extracting videos.')
      extract_archive(path_rar, dir_video)

    # download and extract splits
    path_rar = osp.join(dir_hmdb51, 'test_train_splits.rar')
    dir_split = osp.join(dir_hmdb51, 'testTrainMulti_7030_splits')
    dir_split_processed = osp.join(dir_hmdb51, 'splits')
    if not osp.exists(path_rar):
      print('Downloading splits.')
      wget.download(self.url_split, path_rar)  # may need to download in terminal
    if not osp.isdir(dir_split):
      print('Extracting splits.')
      extract_archive(path_rar, dir_hmdb51)

    os.makedirs(dir_split_processed, exist_ok=True)
    if all([osp.exists(osp.join(dir_split_processed, 'train.csv')),
            osp.exists(osp.join(dir_split_processed, 'test.csv'))]):
      return

    paths_split = glob.glob(osp.join(dir_split, '*_test_split1.txt'))
    cnames = sorted([x.rsplit('/', 1)[1].rsplit('_', 2)[0] for x in paths_split])
    cname_to_cid = {cname:i for i, cname in enumerate(cnames)}

    train, test = [], []
    for cname in cnames:
      data = pd.read_csv(osp.join(dir_split, f'{cname}_test_split1.txt'), header=None, delimiter='\s+')
      train += [[f'{cname}/{x}', cname_to_cid[cname]] for x in data[data[1] == 1][0].values.tolist()]
      test += [[f'{cname}/{x}', cname_to_cid[cname]] for x in data[data[1] == 2][0].values.tolist()]

    with open(osp.join(dir_split_processed, 'train.csv'), 'w') as f:
      writer = csv.writer(f, delimiter=' ')
      for x in train:
        writer.writerow(x)

    with open(osp.join(dir_split_processed, 'test.csv'), 'w') as f:
      writer = csv.writer(f, delimiter=' ')
      for x in test:
        writer.writerow(x)

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)

    if hasattr(self.cfg, 'num_views'):
      clip_sampler = make_clip_sampler('random_multi', self.cfg.T*self.cfg.tau/self.cfg.fps, self.cfg.num_views)
    else:
      clip_sampler = make_clip_sampler('random', self.cfg.T*self.cfg.tau/self.cfg.fps)
    self.dataset_train = MapDataset.from_iterable_dataset(labeled_video_dataset(
      data_path=osp.join(self.cfg.dir_data, 'HMDB-51', 'splits', 'train.csv'),
      clip_sampler=clip_sampler,
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,  # ignored
      transform=transform_train,
      video_path_prefix=osp.join(self.cfg.dir_data, 'HMDB-51', 'videos'),
      decode_audio=False,
      decoder='decord'
    ))

    self.dataset_val = labeled_video_dataset(
      data_path=osp.join(self.cfg.dir_data, 'HMDB-51', 'splits', 'test.csv'),
      clip_sampler=make_clip_sampler('uniform', self.cfg.T*self.cfg.tau/self.cfg.fps),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_val,
      video_path_prefix=osp.join(self.cfg.dir_data, 'HMDB-51', 'videos'),
      decode_audio=False,
      decoder='decord'
    )

    self.dataset_test = labeled_video_dataset(
      data_path=osp.join(self.cfg.dir_data, 'HMDB-51', 'splits', 'test.csv'),
      clip_sampler=make_clip_sampler('constant_clips_per_video', self.cfg.T*self.cfg.tau/self.cfg.fps, 10, 3),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_test,
      video_path_prefix=osp.join(self.cfg.dir_data, 'HMDB-51', 'videos'),
      decode_audio=False,
      decoder='decord'
    )
