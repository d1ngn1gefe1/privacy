from .adapters import get_adapter
from .dp import make_private
from .image_classifier import ImageClassifierModule
from .video_classifier import VideoClassifierModule


def get_model(cfg):
  assert cfg.mode in ['from_scratch', 'fine_tuning', 'linear_probing', 'adapter']
  assert cfg.task in ['multi-class', 'multi-label']

  if cfg.dataset in ['cifar100', 'medmnist', 'chexpert']:
    model = ImageClassifierModule(cfg)
  elif cfg.dataset in ['ucf101']:
    model = VideoClassifierModule(cfg)
  else:
    raise NotImplementedError

  if cfg.mode == 'adapter':
    adapter = get_adapter(cfg)
    model = adapter.convert_adapter(model)

  if cfg.dp:
    model = make_private(model)

  return model
