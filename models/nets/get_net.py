def get_net(net, num_classes, pretrained, dir_weights):
  if net == 'dpsgd_net':
    assert pretrained is False, 'Pre-trained weights not available'
    from .dpsgd_net import get_dpsgd_net
    net = get_dpsgd_net(num_classes)
  elif net == 'opacus_net':
    assert pretrained is False, 'Pre-trained weights not available'
    from .opacus_net import get_opacus_net
    net = get_opacus_net(num_classes)
  elif net == 'vit':
    from .vit import get_vit
    net = get_vit(num_classes, pretrained)
  elif net == 'resnet':
    from .resnet import get_resnet
    net = get_resnet(num_classes, pretrained, dir_weights)
  elif net == 'convnext':
    from .convnext import get_convnext
    net = get_convnext(num_classes, pretrained)
  elif net == 'mvit':
    from .mvit import get_mvit
    net = get_mvit(num_classes, pretrained, dir_weights)
  else:
    raise NotImplementedError

  return net
