from .dpsgd_net import DPSGDNet
from .opacus_net import OpacusNet


def get_net(net, num_classes):
  if net == 'dpsgd_net':
    net = DPSGDNet(num_classes)
  elif net == 'opacus_net':
    net = OpacusNet(num_classes)
  else:
    raise NotImplementedError

  return net
