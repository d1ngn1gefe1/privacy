import csv
from functools import partial
import glob
import json
from multiprocessing import Pool
import os
import os.path as osp
from pytorchvideo.data import labeled_video_dataset, make_clip_sampler, Ucf101
import subprocess
from torch.utils.data import DistributedSampler, RandomSampler

from .base_datamodule import BaseDataModule
from data.transforms import get_transform
from .map_dataset import MapDataset
import utils


def process_video(dir_mp4, dir_avi, name):
  path_mp4 = osp.join(dir_mp4, f'{name}.mp4')
  path_avi = osp.join(dir_avi, f'{name}.avi')
  subprocess.run(['ffmpeg', '-i', path_mp4, '-vcodec', 'mpeg4', '-an', path_avi])


class Diving48DataModule(BaseDataModule):
  def __init__(self, cfg):
    super().__init__(cfg)

    self.cfg.num_classes = 48
    self.cfg.task = 'multi-class'

  def prepare_data(self):
    dir_avi = osp.join(self.cfg.dir_data, 'Diving48', 'videos')
    dir_mp4 = osp.join(self.cfg.dir_data, 'Diving48', 'rgb')
    dir_labels = osp.join(self.cfg.dir_data, 'Diving48', 'labels')
    splits = ['train', 'test']

    # convert .mp4 to .avi to avoid a bug in opacus+ddp
    os.makedirs(dir_avi, exist_ok=True)
    names_mp4 = [osp.splitext(osp.basename(x))[0] for x in glob.glob(osp.join(dir_mp4, '*.mp4'))]
    names_avi = [osp.splitext(osp.basename(x))[0] for x in glob.glob(osp.join(dir_avi, '*.avi'))]
    with Pool(8) as p:
      p.map(partial(process_video, dir_mp4, dir_avi), [x for x in names_mp4 if x not in names_avi])
    names_avi = [osp.splitext(osp.basename(x))[0] for x in glob.glob(osp.join(dir_avi, '*.avi'))]
    assert len(names_mp4) == len(names_avi), f'{len(names_mp4)} vs {len(names_avi)}'

    if not all([osp.exists(osp.join(dir_labels, f'{split}.csv')) for split in splits]):
      for split in splits:
        with open(osp.join(dir_labels, f'Diving48_V2_{split}.json'), 'r') as f:
          data = json.load(f)

        with open(osp.join(dir_labels, f'{split}.csv'), 'w') as f:
          writer = csv.writer(f, delimiter=' ')
          for i, x in enumerate(data):
            writer.writerow([f'{x["vid_name"]}.avi', x['label']])

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)

    dir_videos = osp.join(self.cfg.dir_data, 'Diving48', 'videos')
    dir_labels = osp.join(self.cfg.dir_data, 'Diving48', 'labels')

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
      data_path=osp.join(dir_labels, f'test.csv'),
      clip_sampler=make_clip_sampler('uniform', self.cfg.T*self.cfg.tau/self.cfg.fps),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_val,
      video_path_prefix=dir_videos,
      decode_audio=False
    )

    self.dataset_test = labeled_video_dataset(
      data_path=osp.join(dir_labels, f'test.csv'),
      clip_sampler=make_clip_sampler('constant_clips_per_video', self.cfg.T*self.cfg.tau/self.cfg.fps, 10, 3),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_test,
      video_path_prefix=dir_videos,
      decode_audio=False
    )
