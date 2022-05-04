from colorama import Fore, Back, init
import torch


init(autoreset=True)


def is_ddp():
  return torch.distributed.is_available() and torch.distributed.is_initialized()


def pprint(*args):
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
