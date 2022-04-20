from omegaconf import OmegaConf
import os

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/ucf101/mvit.yaml')
  cfg.phase = 'train'
  cfg.name = utils.get_name(cfg)

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, datamodule=data)


if __name__ == '__main__':
  os.environ['PL_RECONCILE_PROCESS'] = '1'
  os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

  utils.patch()
  main()
