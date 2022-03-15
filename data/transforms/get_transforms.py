from .dpsgd_net import get_dpsgd_net_transforms
from .opacus_net import get_opacus_net_transforms
from .vit import get_vit_transforms


def get_transforms(net, augment):
  if net == 'dpsgd_net':
    return get_dpsgd_net_transforms(augment)
  elif net == 'opacus_net':
    return get_opacus_net_transforms(augment)
  elif net == 'vit':
    return get_vit_transforms(augment)
  else:
    raise NotImplementedError
