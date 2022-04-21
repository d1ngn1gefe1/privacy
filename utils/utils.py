from colorama import Fore, Back, init
from omegaconf.dictconfig import DictConfig
import os
import torch

from .patches import *


init(autoreset=True)


def get_name(cfg):
  name = f'{cfg.dataset}_{cfg.net}_{cfg.mode}_epoch{cfg.num_epochs}_bs{cfg.batch_size}_lr{cfg.lr}_gpu{len(cfg.gpus)}'

  if cfg.dp:
    name += f'_sigma{cfg.sigma}_c{cfg.c}'

  return name


def is_ddp():
  return torch.distributed.is_available() and torch.distributed.is_initialized()


def info(*args):
  if is_ddp():
    args = [f'Rank {torch.distributed.get_rank()}']+list(args)
  print(f'{Fore.WHITE}{Back.RED}{", ".join([str(x) for x in args])}')


def get_type(module):
  text = f'{type(module).__name__}({str(id(module))[-5:]}'
  if hasattr(module, 'parameters') and hasattr(next(module.parameters()), 'grad_sample'):
    text += ',per-sample'
  text += ')'

  if hasattr(module, 'net'):
    text += ' >> [net]'+get_type(module.net)
  if hasattr(module, 'module'):
    text += ' >> [module]'+get_type(module.module)
  return text


def setup(cfg, phase):
  os.environ['PL_RECONCILE_PROCESS'] = '1'

  patch_pytorch_lightning()
  patch_pytorchvideo()
  patch_opacus()

  _parse_cfg(cfg, phase)


def _parse_cfg(cfg, phase):
  cfg.phase = phase
  cfg.name = get_name(cfg)

  if isinstance(cfg.lr, DictConfig):
    cfg.lr = cfg.lr[cfg.optimizer+('_dp' if cfg.dp else '')]

  if isinstance(cfg.wd, DictConfig):
    cfg.wd = cfg.wd[cfg.optimizer]
