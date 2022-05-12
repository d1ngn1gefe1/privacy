from omegaconf.dictconfig import DictConfig

from .patches import patch


def get_name(cfg):
  name = f'{cfg.dataset}_{cfg.net}_{cfg.mode}_{cfg.optimizer}'
  name += f'_epoch{cfg.num_epochs}_bs{cfg.batch_size}_lr{cfg.lr}_gpu{len(cfg.gpus)}'

  if cfg.dp:
    if hasattr(cfg, 'sigma'):
      name += f'_delta{cfg.delta}_sigma{cfg.sigma}_c{cfg.c}'
    else:
      name += f'epsilon{cfg.epsilon}_delta{cfg.delta}_c{cfg.c}'

  return name


def setup(cfg, phase):
  patch(cfg)

  # TODO: fix partial private
  if isinstance(cfg.lr, DictConfig):
    cfg.lr = cfg.lr[cfg.optimizer+('_dp' if cfg.dp else '')]

  if isinstance(cfg.wd, DictConfig):
    cfg.wd = cfg.wd[cfg.optimizer]

  cfg.phase = phase
  cfg.name = get_name(cfg)


def update(cfg, cfg_new):
  cfg.update(cfg_new)
  cfg.name = get_name(cfg)
