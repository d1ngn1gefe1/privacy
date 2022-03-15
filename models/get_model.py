from .dp import make_private
from .image_classifier import ImageClassifierModule


def get_model(cfg):
  assert cfg.mode in ['from_scratch', 'fine_tuning', 'linear_probing']
  model = ImageClassifierModule(cfg)

  if cfg.dp:
    model = make_private(model)

  return model
