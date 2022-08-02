from pytorchvideo_trainer.datamodule.transforms import ApplyTransformToKeyOnList
from torchvision.transforms import Compose
from .transforms import ApplyTransformOnList, Repeat


def get_transform(cfg):
  if cfg.net == 'dpsgd_net':
    from .dpsgd_net import get_dpsgd_net_transforms
    transform_train, transform_val, transform_test = get_dpsgd_net_transforms(cfg)
  elif cfg.net == 'opacus_net':
    from .opacus_net import get_opacus_net_transforms
    transform_train, transform_val, transform_test = get_opacus_net_transforms(cfg)
  elif cfg.net == 'vit':
    from .vit import get_vit_transforms
    transform_train, transform_val, transform_test = get_vit_transforms(cfg)
  elif cfg.net.startswith('resnet'):
    from .resnet import get_resnet_transforms
    transform_train, transform_val, transform_test = get_resnet_transforms(cfg)
  elif cfg.net == 'convnext':
    from .convnext import get_convnext_transforms
    transform_train, transform_val, transform_test = get_convnext_transforms(cfg)
  elif cfg.net == 'mvit':
    from .mvit import get_mvit_transforms
    transform_train, transform_val, transform_test = get_mvit_transforms(cfg)
  else:
    raise NotImplementedError

  if hasattr(cfg, 'num_views'):
    if cfg.net == 'mvit':
      transform_train = Compose(transforms=[
        ApplyTransformToKeyOnList(key='video', transform=transform_train.transforms[0]._transform),
        transform_train.transforms[1]
      ])
    else:
      transform_train = Compose([
        Repeat(cfg.num_views),
        ApplyTransformOnList(transform=transform_train)
      ])

  return transform_train, transform_val, transform_test
