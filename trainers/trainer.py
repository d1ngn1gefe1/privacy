import os

from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
#from ray_lightning import RayPlugin
#from ray_lightning.tune import TuneReportCallback

from .callbacks import DPCallback


def get_trainer(cfg, trial=None):
  # logger
  logger = WandbLogger(
    project=cfg.dataset,
    name=cfg.name,
    log_model='all',
    save_dir=cfg.dir_log+f'_{os.getlogin()}'
  )
  logger.log_hyperparams(cfg)

  # callbacks and plugins
  callbacks = [
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(every_n_epochs=5,
                    save_last=True,
                    dirpath=os.path.join(cfg.dir_weights, f'ckpt_{os.getlogin()}/{cfg.name}'))
  ]
  plugins = None
  if cfg.phase == 'tune':
    #callbacks.append(TuneReportCallback(metrics={'acc': 'val/acc'}, on='validation_end'))
    #plugins = [RayPlugin(num_workers=2, num_cpus_per_worker=cfg.num_workers, use_gpu=True)]
    callbacks.append(PyTorchLightningPruningCallback(trial, monitor='val/acc'))
  if cfg.dp:
    callbacks.append(DPCallback())

  # all other kwargs
  kwargs = {
    'max_epochs': cfg.num_epochs,
    'logger': logger,
    'callbacks': callbacks,
    'plugins': plugins,
    'check_val_every_n_epoch': 1,
    'num_sanity_val_steps': 2,
    'log_every_n_steps': 10,
    'accelerator': 'gpu',
    'devices': cfg.gpus,
    'strategy': DDPStrategy(find_unused_parameters=False) if len(cfg.gpus) > 1 else None,
    'detect_anomaly': True
  }

  trainer = Trainer(**kwargs)
  return trainer
