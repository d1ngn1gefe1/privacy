from omegaconf import OmegaConf

from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
import pytorch_lightning as pl
import pytorch_lightning.overrides.base
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase
from torch.nn import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP


# TODO 1
def ssss(wrapped_model):
  model = wrapped_model
  if isinstance(model, (DPDDP, DDP, DP)):
    model = ssss(model.module)
  if isinstance(model, (_LightningModuleWrapperBase, _LightningPrecisionModuleWrapperBase)):
    model = ssss(model.module)
  if not isinstance(model, pl.LightningModule):
    raise TypeError(f"Unwrapping the module did not yield a `LightningModule`, got {type(model)} instead.")
  return model
pytorch_lightning.overrides.base.unwrap_lightning_module = ssss

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/opacus_net.yaml')
  cfg.phase = 'train'
  cfg.name = utils.get_name(cfg)

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, datamodule=data)


if __name__ == '__main__':
  main()

