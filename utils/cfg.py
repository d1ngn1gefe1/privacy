from omegaconf.dictconfig import DictConfig

from .patches import patch


def get_name(cfg):
  name = f'{cfg.dataset}_{cfg.net}_{cfg.mode}_{cfg.optimizer}'
  name += f'_epoch{cfg.num_epochs}_bs{cfg.batch_size["train"]}_lr{cfg.lr}_gpu{len(cfg.gpus)}'

  if hasattr(cfg, 'num_views'):
    name += f'_view{cfg.num_views}'

  if cfg.dp:
    if hasattr(cfg, 'sigma'):
      name += f'_delta{cfg.delta}_sigma{cfg.sigma}_c{cfg.c}'
    else:
      name += f'_epsilon{cfg.epsilon}_delta{cfg.delta}_c{cfg.c}'

  return name


def setup(cfg, phase):
  patch(cfg)

  # TODO: fix partial private
  if isinstance(cfg.lr, DictConfig):
    cfg.lr = cfg.lr[cfg.optimizer+('_dp' if cfg.dp else '')]

  if isinstance(cfg.wd, DictConfig):
    cfg.wd = cfg.wd[cfg.optimizer]

  if not isinstance(cfg.batch_size, DictConfig):
    cfg.batch_size = DictConfig({'train': cfg.batch_size, 'val': cfg.batch_size, 'test': cfg.batch_size})

  cfg.phase = phase
  cfg.name = get_name(cfg)


def update(cfg, cfg_new):
  cfg.update(cfg_new)
  cfg.name = get_name(cfg)
