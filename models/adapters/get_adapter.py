from .mlp import MlpAdapter
from .bottleneck import BottleneckAdapter
from .video_mlp import VideoMlpAdapter
from .mlp_clip import MlpClipAdapter


def get_adapter(cfg):
  if cfg.adapter == 'mlp_adapter':
    adapter = MlpAdapter
  elif cfg.adapter == 'bottleneck_adapter':
    adapter = BottleneckAdapter
  elif cfg.adapter == 'video_mlp_adapter':
    adapter = VideoMlpAdapter
  elif cfg.adapter == 'mlp_clip_adapter':
    adapter = MlpClipAdapter
  else:
    raise NotImplementedError

  return adapter
