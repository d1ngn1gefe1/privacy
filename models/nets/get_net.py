def get_net(cfg):
  if cfg.net == 'dpsgd_net':
    from .dpsgd_net import get_dpsgd_net
    net = get_dpsgd_net(cfg)
  elif cfg.net == 'opacus_net':
    from .opacus_net import get_opacus_net
    net = get_opacus_net(cfg)
  elif cfg.net == 'vit':
    from .vit import get_vit
    net = get_vit(cfg)
  elif cfg.net == 'resnet':
    from .resnet import get_resnet
    net = get_resnet(cfg)
  elif cfg.net == 'convnext':
    from .convnext import get_convnext
    net = get_convnext(cfg)
  elif cfg.net == 'mvit':
    from .mvit import get_mvit
    net = get_mvit(cfg)
  else:
    raise NotImplementedError

  return net
