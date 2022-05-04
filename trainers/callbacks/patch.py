from pytorch_lightning.callbacks.base import Callback

from utils import patch


class PatchCallback(Callback):
  """ Need another patching for ddp_spawn (multiprocessing)
  """
  def setup(self, trainer, pl_module, stage):
    patch()
