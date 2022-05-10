def get_transform(cfg):
  if cfg.net == 'dpsgd_net':
    from .dpsgd_net import get_dpsgd_net_transforms
    return get_dpsgd_net_transforms(cfg.augment)
  elif cfg.net == 'opacus_net':
    from .opacus_net import get_opacus_net_transforms
    return get_opacus_net_transforms(cfg.augment)
  elif cfg.net == 'vit':
    from .vit import get_vit_transforms
    return get_vit_transforms(cfg.augment)
  elif cfg.net == 'resnet':
    from .resnet import get_resnet_transforms
    return get_resnet_transforms(cfg.augment)
  elif cfg.net == 'convnext':
    from .convnext import get_convnext_transforms
    return get_convnext_transforms(cfg.augment)
  elif cfg.net == 'mvit':
    from .mvit import get_mvit_transforms
    return get_mvit_transforms(cfg.augment)
  else:
    raise NotImplementedError
