from omegaconf import OmegaConf

from data import get_data
from models import get_model
from trainers import get_trainer


def main():
  #cfg = OmegaConf.load('./configs/config.yaml')
  cfg = OmegaConf.load('./configs/config_dp.yaml')

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, data)


if __name__ == '__main__':
  main()

