from models.adapters import get_adapter
from models.modules.image_classifier import ImageClassifierModule
from models.modules.video_classifier import VideoClassifierModule


def get_model(cfg):
  assert cfg.mode in ['from_scratch', 'fine_tuning', 'linear_probing', 'adapter']
  assert cfg.task in ['multi-class', 'multi-label']

  if cfg.dataset in ['cifar100', 'cifar10', 'medmnist', 'chexpert', 'imagenet']:
    model = ImageClassifierModule(cfg)
  elif cfg.dataset in ['ucf101']:
    model = VideoClassifierModule(cfg)
  else:
    raise NotImplementedError

  if cfg.mode == 'adapter':
    adapter = get_adapter(cfg)
    model = adapter.convert_adapter(model)

  return model
