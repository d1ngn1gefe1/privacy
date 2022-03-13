import os

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
#from pytorch_lightning.strategies import DDPStrategy


def get_trainer(cfg):
  name = 'non_dp_cifar10'
  logger = WandbLogger(
    project='cifar10',
    name=name,
    log_model='all',
    save_dir=cfg.dir_wandb
  )
  logger.log_hyperparams(cfg)

  trainer = Trainer(
    max_epochs=cfg.num_epochs,
    logger=logger,
    callbacks=[
      ModelCheckpoint(every_n_epochs=5,
                      save_last=True,
                      dirpath=os.path.join(cfg.dir_weights, f'ckpt/{name}'))
    ],
    check_val_every_n_epoch=1,
    precision=16,
    log_every_n_steps=100,
    gpus=cfg.gpus,
    **(
      {
        #'strategy': DDPStrategy(find_unused_parameters=False)
        'strategy': 'ddp'
      } if len(cfg.gpus) > 1 else {}
    )
  )

  return trainer
