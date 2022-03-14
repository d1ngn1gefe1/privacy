from .image_classifier import ImageClassifierModule
from .lightning import DPLightningModule


def get_model(cfg):
  model = ImageClassifierModule(cfg)
  if cfg.dp:
    model = DPLightningModule(model)
  return model
