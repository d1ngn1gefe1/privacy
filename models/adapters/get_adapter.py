from .mlp import MlpAdapter
from .bottleneck import BottleneckAdapter


def get_adapter(cfg):
  if cfg.adapter == 'mlp_adapter':
    adapter = MlpAdapter
  elif cfg.adapter == 'bottleneck_adapter':
    adapter = BottleneckAdapter
  else:
    raise NotImplementedError

  return adapter
