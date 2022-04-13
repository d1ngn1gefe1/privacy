from colorama import Fore, Back, init

from .patches import *


init(autoreset=True)


def patch():
  patch_lightning()
  patch_opacus()


def get_name(cfg):
  name = f'{cfg.dataset}_{cfg.net}_{cfg.mode}_epoch{cfg.num_epochs}_bs{cfg.batch_size}_lr{cfg.lr}_gpu{len(cfg.gpus)}'

  if cfg.dp:
    name += f'_sigma{cfg.sigma}_c{cfg.c}'

  if cfg.phase == 'tune':
    name += '_tune'

  return name


def info(text):
  print(Fore.WHITE+Back.RED+text)


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
