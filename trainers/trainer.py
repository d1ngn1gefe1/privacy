import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from ray.tune.integration.pytorch_lightning import TuneReportCallback


def get_trainer(cfg):
  name = f'{cfg.dataset}_{cfg.net}_epoch{cfg.num_epochs}_bs{cfg.batch_size}_lr{cfg.lr}'
  if cfg.dp:
    name += f'_dp_sigma{cfg.sigma}_c{cfg.c}'
  else:
    name += f'_non_dp'
  logger = WandbLogger(
    project='cifar10',
    name=name,
    log_model='all',
    save_dir=cfg.dir_log+f'_{os.getlogin()}'
  )
  logger.log_hyperparams(cfg)

  callbacks = [
    LearningRateMonitor(logging_interval='step'),
    ModelCheckpoint(every_n_epochs=5,
                    save_last=True,
                    dirpath=os.path.join(cfg.dir_weights, f'ckpt_{os.getlogin()}/{name}'))
  ]
  if cfg.tune:
    metrics = {'acc': 'val/acc'}
    callbacks.append(TuneReportCallback(metrics, on='validation_end'))
  trainer = Trainer(
    max_epochs=cfg.num_epochs,
    logger=logger,
    callbacks=callbacks,
    check_val_every_n_epoch=1,
    num_sanity_val_steps=2,
    log_every_n_steps=50,
    gpus=cfg.gpus,
    **(
      {
        'strategy': DDPStrategy(find_unused_parameters=False)
      } if len(cfg.gpus) > 1 else {}
    )
  )

  return trainer
