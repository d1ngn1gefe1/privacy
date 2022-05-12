from omegaconf import OmegaConf

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  # cfg = OmegaConf.load('configs/ucf101/mvit.yaml')
  cfg = OmegaConf.load('configs/cifar100/vit.yaml')
  utils.setup(cfg, 'fit')

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, datamodule=data)


if __name__ == '__main__':
  main()
