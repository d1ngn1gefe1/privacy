from omegaconf import OmegaConf

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
  cfg.phase = 'train'
  cfg.name = utils.get_name(cfg)

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, data)


if __name__ == '__main__':
  main()

