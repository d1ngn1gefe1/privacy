import csv
import glob
import json
import os.path as osp
from pytorchvideo.data import labeled_video_dataset, make_clip_sampler
from torch.utils.data import DistributedSampler, RandomSampler

from .base_datamodule import BaseDataModule
from data.transforms import get_transform
from .map_dataset import MapDataset
import utils


class SSv2DataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = 174
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    dir_ssv2 = osp.join(self.cfg.dir_data, 'ssv2')
    dir_videos = osp.join(dir_ssv2, 'videos')
    dir_labels = osp.join(dir_ssv2, 'labels')

    names_video = [osp.splitext(osp.basename(x))[0] for x in glob.glob(osp.join(dir_videos, '*.webm'))]
    assert len(names_video) == 220847

    splits = ['train', 'validation']
    if all([osp.exists(osp.join(dir_labels, f'{split}.csv')) for split in splits]):
      return

    with open(osp.join(dir_labels, 'labels.json'), 'r') as f:
      labels = json.load(f)

    for split in splits:
      with open(osp.join(dir_labels, f'{split}.json'), 'r') as f:
        data = json.load(f)

      with open(osp.join(dir_labels, f'{split}.csv'), 'w') as f:
        writer = csv.writer(f, delimiter=' ')

        for x in data:
          template = x['template'].replace('[', '').replace(']', '')
          label = labels[template]
          path = f'{x["id"]}.webm'
          writer.writerow([path, label])

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)

    dir_labels = osp.join(self.cfg.dir_data, 'ssv2', 'labels')
    dir_videos = osp.join(self.cfg.dir_data, 'ssv2', 'videos')

    if hasattr(self.cfg, 'num_views'):
      clip_sampler = make_clip_sampler('random_multi', self.cfg.T*self.cfg.tau/self.cfg.fps, self.cfg.num_views)
    else:
      clip_sampler = make_clip_sampler('random', self.cfg.T*self.cfg.tau/self.cfg.fps)

    self.dataset_train = MapDataset.from_iterable_dataset(labeled_video_dataset(
      data_path=osp.join(dir_labels, f'train.csv'),
      clip_sampler=clip_sampler,
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,  # ignored
      transform=transform_train,
      video_path_prefix=dir_videos,
      decode_audio=False
    ))

    self.dataset_val = labeled_video_dataset(
      data_path=osp.join(dir_labels, f'validation.csv'),
      clip_sampler=make_clip_sampler('uniform', self.cfg.T*self.cfg.tau/self.cfg.fps),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_val,
      video_path_prefix=dir_videos,
      decode_audio=False
    )

    self.dataset_test = labeled_video_dataset(
      data_path=osp.join(dir_labels, f'validation.csv'),
      clip_sampler=make_clip_sampler('constant_clips_per_video', self.cfg.T*self.cfg.tau/self.cfg.fps, 10, 3),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_test,
      video_path_prefix=dir_videos,
      decode_audio=False
    )
