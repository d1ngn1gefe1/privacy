from .img_cls import ImgClsModule
from .lightning import DPLightningModule


def get_model(cfg):
  model = ImgClsModule(cfg)
  if cfg.dp:
    model = DPLightningModule(model)
  return model

