from functools import partial
from omegaconf import OmegaConf
import optuna
import os
import os.path as osp

from data import get_data
from models import get_model
from trainers import get_trainer
import utils


def main():
  cfg = OmegaConf.load('configs/cifar100/resnet.yaml')
  cfg_tune = OmegaConf.load('configs/tune.yaml')
  utils.setup(cfg, 'tune')

  if osp.exists(cfg_tune.db):
    os.remove(cfg_tune.db)

  storage = f'sqlite:///{cfg_tune.db}'
  pruner = optuna.pruners.HyperbandPruner()
  sampler = optuna.samplers.TPESampler()
  study = optuna.create_study(study_name=f'{cfg.dataset}_{cfg.net}', storage=storage, direction='maximize',
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
  # update cfg
  cfg_tune = OmegaConf.to_container(cfg_tune)
  cfg_tune.pop('db')

  cfg_sample = {}
  for k, v in cfg_tune.items():
    kind = v.pop('kind')
    cfg_sample[k] = getattr(trial, f'suggest_{kind}')(k, **v)

  utils.update(cfg, cfg_sample)
  print(cfg.name)

  # fit
  data = get_data(cfg)
  model = get_model(cfg)
  trainer = get_trainer(cfg, trial)

  trainer.fit(model, datamodule=data)
  return trainer.callback_metrics['val/acc'].item()


if __name__ == '__main__':
  main()
