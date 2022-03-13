from .dpsgd_net import DPSGDNet


def get_net(cfg):
  if cfg.net == 'dpsgd_net':
    net = DPSGDNet(cfg.num_classes)
  else:
    raise NotImplementedError

  return net

