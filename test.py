from omegaconf import OmegaConf
import os

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/opacus_net.yaml')
  cfg.phase = 'test'
  cfg.name = utils.get_name(cfg)

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  path_ckpt = os.path.join(cfg.dir_weights, cfg.relpath_ckpt)
  trainer.test(model, datamodule=data, ckpt_path=path_ckpt)


if __name__ == '__main__':
  utils.setup()
  main()
