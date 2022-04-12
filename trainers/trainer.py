import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from .dp_callback import DPCallback


def get_trainer(cfg):
  logger = WandbLogger(
    project=cfg.dataset,
    name=cfg.name,
    log_model='all',
    save_dir=cfg.dir_log+f'_{os.getlogin()}'
  )
  logger.log_hyperparams(cfg)

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

  kwargs = {
    'max_epochs': cfg.num_epochs,
    'logger': logger,
    'callbacks': callbacks,
    'check_val_every_n_epoch': 1,
    'num_sanity_val_steps': 0,
    'log_every_n_steps': 50,
    'gpus': cfg.gpus,
    'replace_sampler_ddp': False
  }
  if len(cfg.gpus) > 1:
    kwargs['strategy'] = DDPStrategy(find_unused_parameters=False)

  trainer = Trainer(**kwargs)

  return trainer
