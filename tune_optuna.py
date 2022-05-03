from omegaconf import OmegaConf
import optuna

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar10/opacus_net.yaml')
  utils.setup(cfg, 'tune')

  cfg_tune = OmegaConf.load('configs/tune.yaml')

  storage = 'sqlite:///example.db'
  pruner = optuna.pruners.MedianPruner()
  study = optuna.create_study(study_name=cfg.name, storage=storage, direction='maximize', pruner=pruner,
                              load_if_exists=True)
  objective = lambda x: train(x, cfg_tune, cfg)
  study.optimize(objective, n_trials=3, timeout=600)

  print(f'Number of finished trials: {len(study.trials)}')

  print('Best trial:')
  trial = study.best_trial

  print(f'  Value: {trial.value}')

  print('  Params: ')
  for key, value in trial.params.items():
    print(f'    {key}: {value}')


def train(trial, cfg_tune, cfg):
  cfg_tune = OmegaConf.to_container(cfg_tune)
  cfg_tune = {k: trial.suggest_categorical(k, v) for k, v in cfg_tune.items()}
  cfg.update(cfg_tune)  # overwrite cfg

  print(cfg)
  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg, trial)

  trainer.fit(model, datamodule=data)
  return trainer.callback_metrics['val/acc'].item()


if __name__ == '__main__':
  main()
