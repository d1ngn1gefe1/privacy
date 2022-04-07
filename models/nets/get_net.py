from .dpsgd_net import get_dpsgd_net
from .opacus_net import get_opacus_net
from .vit import get_vit
from .resnet import get_resnet
from .convnext import get_convnext
from .mvit import get_mvit


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
  elif net == 'convnext':
    net = get_convnext(num_classes, pretrained)
  elif net == 'mvit':
    net = get_mvit(num_classes, pretrained)
  else:
    raise NotImplementedError

  return net
