from pytorch_lightning.callbacks.base import Callback

from utils import patch


class PatchCallback(Callback):
  def setup(self, trainer, pl_module, stage):
    patch()
