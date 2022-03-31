from omegaconf import OmegaConf
from ray import tune

from data import get_data
from models import get_model
from trainers import get_trainer


def main():
  cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
  if not cfg.tune:
    train(cfg)
  else:
    cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
    # Replace default hyper with search range
    cfg.num_epochs = tune.choice(cfg.tune_epoch)
    cfg.lr = tune.choice(cfg.tune_lr)
    cfg.batch_size = tune.choice(cfg.tune_batch_size)
    cfg.sigma = tune.choice(cfg.tune_sigma)
    cfg.c = tune.choice(cfg.tune_c)

    trainable = tune.with_parameters(train)
    analysis = tune.run(
      trainable,
      resources_per_trial={
        'cpu': 1,
        'gpu': len(cfg.gpus)
      },
      metric='acc',
      mode='max',
      config=cfg,
      num_samples=10,  # how many trials
      name=f'tune_{cfg.dataset}'
    )


def train(cfg):
  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg)

  trainer.fit(model, data)


if __name__ == '__main__':
  main()

