def get_transform(net, augment):
  if net == 'dpsgd_net':
    from .dpsgd_net import get_dpsgd_net_transforms
    return get_dpsgd_net_transforms(augment)
  elif net == 'opacus_net':
    from .opacus_net import get_opacus_net_transforms
    return get_opacus_net_transforms(augment)
  elif net == 'vit':
    from .vit import get_vit_transforms
    return get_vit_transforms(augment)
  elif net == 'resnet':
    from .resnet import get_resnet_transforms
    return get_resnet_transforms(augment)
  elif net == 'convnext':
    from .convnext import get_convnext_transforms
    return get_convnext_transforms(augment)
  elif net == 'mvit':
    from .mvit import get_mvit_transforms
    return get_mvit_transforms(augment)
  else:
    raise NotImplementedError
