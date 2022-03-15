from .dp import make_private
from .image_classifier import ImageClassifierModule


def get_model(cfg):
  model = ImageClassifierModule(cfg)

  if cfg.dp:
    model = make_private(model)

  return model
