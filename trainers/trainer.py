import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from .callbacks import DPCallback


def get_trainer(cfg):
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
    ModelCheckpoint(every_n_epochs=5,
                    save_last=True,
                    dirpath=os.path.join(cfg.dir_weights, f'ckpt_{os.getlogin()}/{cfg.name}'))
  ]
  if cfg.phase == 'tune':
    callbacks.append(TuneReportCallback({'acc': 'val/acc', 'acc-div-log_epsilon': 'val/acc-div-log_epsilon'}, on='validation_end'))
  if cfg.dp:
    callbacks.append(DPCallback())

  # all other kwargs
  kwargs = {
    'max_epochs': cfg.num_epochs,
    'logger': logger,
    'callbacks': callbacks,
    'check_val_every_n_epoch': 1,
    'num_sanity_val_steps': 2,
    'log_every_n_steps': 10,
    'accelerator': 'gpu',
    'devices': cfg.gpus,
    'strategy': DDPStrategy() if len(cfg.gpus) > 1 else None,
    'detect_anomaly': True
  }

  trainer = Trainer(**kwargs)
  return trainer
