from .adapters import get_adapter
from .dp import make_private
from .image_classifier import ImageClassifierModule


def get_model(cfg):
  assert cfg.mode in ['from_scratch', 'fine_tuning', 'linear_probing', 'adapter']
  model = ImageClassifierModule(cfg)
  if cfg.mode == 'adapter':
    adapter = get_adapter(cfg)
    model = adapter.convert_adapter(model)

  if cfg.dp:
    model = make_private(model)

  return model
