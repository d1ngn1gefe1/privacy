from models.adapters import get_adapter
from models.modules.image_classifier import ImageClassifierModule
from models.modules.video_classifier import VideoClassifierModule


def get_model(cfg):
  assert cfg.mode in ['from_scratch', 'full_tuning', 'linear_probing', 'adapter', 'sparse_tuning']
  assert cfg.task in ['multi-class', 'multi-label']

  if cfg.dataset in ['cifar100', 'cifar10', 'medmnist', 'chexpert', 'imagenet', 'places365']:
    model = ImageClassifierModule(cfg)
  elif cfg.dataset in ['ucf101']:
    model = VideoClassifierModule(cfg)
  else:
    raise NotImplementedError

  # TODO: move inside base_classifer.py
  if cfg.mode == 'adapter':
    adapter = get_adapter(cfg)
    model = adapter.convert_adapter(model)

  return model
