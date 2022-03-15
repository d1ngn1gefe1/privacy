import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin


def get_trainer(cfg):
  name = f"cifar10_{'dp' if cfg.dp else 'non_dp'}"
  logger = WandbLogger(
    project='cifar10',
    name=name,
    log_model='all',
    save_dir=cfg.dir_log+f'_{os.getlogin()}'
  )
  logger.log_hyperparams(cfg)

  trainer = Trainer(
    max_epochs=cfg.num_epochs,
    logger=logger,
    callbacks=[
      ModelCheckpoint(every_n_epochs=5,
                      save_last=True,
                      dirpath=os.path.join(cfg.dir_weights, f'ckpt_{os.getlogin()}/{name}'))
    ],
    check_val_every_n_epoch=1,
    num_sanity_val_steps=2,
    log_every_n_steps=50,
    gpus=cfg.gpus,
    **(
      {
        'strategy': DDPPlugin(find_unused_parameters=False)
      } if len(cfg.gpus) > 1 else {}
    )
  )

  return trainer
