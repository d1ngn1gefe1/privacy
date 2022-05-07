from omegaconf import OmegaConf
import os.path as osp

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/opacus_net.yaml')
  utils.setup(cfg, 'test')

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  path_ckpt = osp.join(cfg.dir_weights, cfg.relpath_ckpt)
  trainer.test(model, datamodule=data, ckpt_path=path_ckpt)


if __name__ == '__main__':
  main()
