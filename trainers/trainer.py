import os

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, DDPSpawnStrategy

from .callbacks import DPCallback, PatchCallback


def get_trainer(cfg, trial=None):
  # logger
  logger = WandbLogger(
    project=cfg.dataset,
    name=cfg.name,
    log_model='all',
    save_dir=cfg.dir_log+f'_{os.getlogin()}'
  )
  logger.log_hyperparams(cfg)

  # callbacks
  callbacks = [
    LearningRateMonitor(logging_interval='step'),
    PatchCallback()
  ]
  if cfg.phase == 'tune':
    callbacks.append(PyTorchLightningPruningCallback(trial, monitor='val/acc'))
  else:
    callbacks.append(ModelCheckpoint(every_n_epochs=5, save_last=True,
                                     dirpath=os.path.join(cfg.dir_weights, f'ckpt_{os.getlogin()}/{cfg.name}')))
  if cfg.dp:
    callbacks.append(DPCallback())

  # strategy
  if len(cfg.gpus) > 1:
    strategy = DDPSpawnStrategy(find_unused_parameters=False) if cfg.phase == 'tune' else \
               DDPStrategy(find_unused_parameters=False)
  else:
    strategy = None

  # all other kwargs
  kwargs = {
    'max_epochs': cfg.num_epochs,
    'logger': logger,
    'callbacks': callbacks,
    'enable_checkpointing': cfg.phase != 'tune',
    'check_val_every_n_epoch': 1,
    'num_sanity_val_steps': 2,
    'log_every_n_steps': 10,
    'accelerator': 'gpu',
    'devices': cfg.gpus,
    'strategy': strategy,
    'detect_anomaly': True
  }

  trainer = Trainer(**kwargs)
  return trainer
