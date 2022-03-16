from .dpsgd_net import get_dpsgd_net
from .opacus_net import get_opacus_net
from .vit import get_vit
from .resnet import get_resnet


def get_net(net, num_classes, pretrained, dir_weights):
  if net == 'dpsgd_net':
    assert pretrained is False, 'Pre-trained weights not available'
    net = get_dpsgd_net(num_classes)
  elif net == 'opacus_net':
    assert pretrained is False, 'Pre-trained weights not available'
    net = get_opacus_net(num_classes)
  elif net == 'vit':
    net = get_vit(num_classes, pretrained)
  elif net == 'resnet':
    net = get_resnet(num_classes, pretrained, dir_weights)
  else:
    raise NotImplementedError

  return net
