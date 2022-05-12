from .pytorch_lightning import patch_pytorch_lightning
from .pytorchvideo import patch_pytorchvideo
from .opacus import patch_opacus
from .optuna import patch_optuna


def patch(cfg):
  patch_pytorch_lightning()
  patch_pytorchvideo()
  patch_opacus(cfg.num_views if hasattr(cfg, 'num_views') else None)
  patch_optuna()
