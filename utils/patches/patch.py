from .pytorch_lightning import patch_pytorch_lightning
from .pytorchvideo import patch_pytorchvideo
from .opacus import patch_opacus
from .optuna import patch_optuna


def patch():
  patch_pytorch_lightning()
  patch_pytorchvideo()
  patch_opacus()
  patch_optuna()
