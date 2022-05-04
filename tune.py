from functools import partial
from omegaconf import OmegaConf
import optuna

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
  utils.setup(cfg, 'tune')
  cfg_tune = OmegaConf.load('configs/tune.yaml')

  storage = 'sqlite:///optuna.db'
  pruner = optuna.pruners.HyperbandPruner()
  sampler = optuna.samplers.TPESampler()
  study = optuna.create_study(study_name=cfg.name, storage=storage, direction='maximize', load_if_exists=True,
                              pruner=pruner, sampler=sampler)
  study.optimize(partial(objective, cfg=cfg, cfg_tune=cfg_tune), n_trials=100, timeout=600)

  print(f'Number of finished trials: {len(study.trials)}')

  print('Best trial:')
  trial = study.best_trial

  print(f'  Value: {trial.value}')

  print('  Params: ')
  for key, value in trial.params.items():
    print(f'    {key}: {value}')


def objective(trial, cfg, cfg_tune):
  cfg_tune = OmegaConf.to_container(cfg_tune)

  cfg_sampled = {}
  # for k, v in cfg_tune.items():
  #   kind = v.pop('kind')
  #   cfg_sampled[k] = getattr(trial, f'suggest_{kind}')(k, **v)

  a = trial.suggest_int('batch_size', 512, 2048)  # todo: debug this line
  assert False

  cfg.update(cfg_sampled)

  print(cfg)
  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg, trial)

  trainer.fit(model, datamodule=data)
  return trainer.callback_metrics['val/acc'].item()


if __name__ == '__main__':
  main()
