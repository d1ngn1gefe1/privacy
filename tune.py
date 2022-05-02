from omegaconf import OmegaConf
from ray import tune

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
  utils.setup(cfg, 'tune')

  cfg_tune = OmegaConf.load('configs/tune.yaml')
  cfg_tune = OmegaConf.to_container(cfg_tune)
  cfg_tune = {k:tune.choice(v) for k, v in cfg_tune.items()}

  trainable = tune.with_parameters(train, cfg=cfg)
  scheduler = None  # todo
  reporter = None  # todo

  analysis = tune.run(
    trainable,
    resources_per_trial={
      'cpu': 1,
      'gpu': len(cfg.gpus)
    },
    metric='acc',
    mode='max',
    config=cfg_tune,
    num_samples=20,  # number of trials
    scheduler=scheduler,
    progress_reporter=reporter,
    name=cfg.name
  )

  print(f'Best hyperparameters: {analysis.best_config}')


def train(cfg_tune, cfg):
  cfg.update(cfg_tune)  # overwrite cfg

  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, datamodule=data)


if __name__ == '__main__':
  main()
