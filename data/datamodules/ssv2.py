import glob
import os
import os.path as osp
from pytorchvideo.data import SSv2, make_clip_sampler
import subprocess
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
    dir_frames = osp.join(dir_ssv2, 'frames')

    names_video = [osp.splitext(osp.basename(x))[0] for x in glob.glob(osp.join(dir_videos, '*.webm'))]
    assert len(names_video) == 220847
    if osp.isdir(dir_frames) and all([osp.isdir(osp.join(dir_frames, name)) for name in names_video]):
      return

    print('Extracting frames.')
    os.makedirs(dir_frames, exist_ok=True)
    for name in names_video:
      os.makedirs(osp.join(dir_frames, name), exist_ok=True)
      path_in = osp.join(dir_videos, f'{name}.webm')
      path_out = osp.join(dir_frames, name, f'{name}_%06d.jpg')
      subprocess.call(['ffmpeg', '-i', path_in, '-r', '30', '-q:v', '1', path_out])

  def setup(self, stage=None):
    transform_train, transform_val, transform_test = get_transform(self.cfg)

    if hasattr(self.cfg, 'num_views'):
      clip_sampler = make_clip_sampler('random_multi', self.cfg.T*self.cfg.tau/self.cfg.fps, self.cfg.num_views)
    else:
      clip_sampler = make_clip_sampler('random', self.cfg.T*self.cfg.tau/self.cfg.fps)
    self.dataset_train = MapDataset.from_iterable_dataset(SSv2(
      label_name_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'labels.csv'),
      video_label_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'train.json'),
      video_path_label_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'train.csv'),
      clip_sampler=clip_sampler,
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,  # ignored
      transform=transform_train,
      video_path_prefix=osp.join(self.cfg.dir_data, 'ssv2', 'frames')
    ))

    self.dataset_val = SSv2(
      label_name_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'labels.csv'),
      video_label_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'validation.json'),
      video_path_label_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'val.csv'),
      clip_sampler=make_clip_sampler('uniform', self.cfg.T*self.cfg.tau/self.cfg.fps),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_val,
      video_path_prefix=osp.join(self.cfg.dir_data, 'ssv2', 'frames')
    )

    self.dataset_test = SSv2(
      label_name_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'labels.csv'),
      video_label_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'validation.json'),
      video_path_label_file=osp.join(self.cfg.dir_data, 'ssv2', 'labels', 'val.csv'),
      clip_sampler=make_clip_sampler('constant_clips_per_video', self.cfg.T*self.cfg.tau/self.cfg.fps, 10, 3),
      video_sampler=DistributedSampler if utils.is_ddp() else RandomSampler,
      transform=transform_test,
      video_path_prefix=osp.join(self.cfg.dir_data, 'ssv2', 'frames')
    )
