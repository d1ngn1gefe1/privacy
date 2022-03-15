from .dpsgd_net import get_dpsgd_net
from .opacus_net import get_opacus_net
from .vit import get_vit


def get_net(net, num_classes, pretrained):
  if net == 'dpsgd_net':
    net = get_dpsgd_net(num_classes, pretrained)
  elif net == 'opacus_net':
    net = get_opacus_net(num_classes, pretrained)
  elif net == 'vit':
    net = get_vit(num_classes, pretrained)
  else:
    raise NotImplementedError

  return net