from omegaconf import OmegaConf

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar10/opacus_net.yaml')
  cfg.phase = 'train'
  cfg.name = utils.get_name(cfg)

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, datamodule=data)


if __name__ == '__main__':
  utils.patch()
  main()
