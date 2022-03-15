from .utils import MlpAdapter


def get_adapter(cfg):
  if cfg.adapter == 'mlp_adapter':
    adapter = MlpAdapter
  else:
    raise NotImplementedError

  return adapter
