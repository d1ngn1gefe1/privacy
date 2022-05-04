from opacus.grad_sample import register_grad_sampler
import os
import timm
from timm.models.layers.norm import GroupNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets.utils import download_url
from types import MethodType
from typing import Dict


@register_grad_sampler(GroupNorm)
def compute_group_norm_timm_grad_sample(
    layer: GroupNorm, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  gs = F.group_norm(activations, layer.num_groups, layer.weight, layer.bias, eps=layer.eps)*backprops
  ret = {
    layer.weight: torch.einsum('ni...->ni', gs),
    layer.bias: torch.einsum('ni...->ni', backprops)
  }
  return ret


def get_classifier(self):
  return self.fc


def get_resnet(num_classes, pretrained, dir_weights, implementation='timm'):
  assert implementation in ['ppwwyyxx', 'timm']
  if implementation == 'ppwwyyxx':
    net = models.__dict__['resnet50'](pretrained=False, num_classes=num_classes,
                                      norm_layer=(lambda x: nn.GroupNorm(32, x)))

    net.get_classifier = MethodType(get_classifier, net)

    if pretrained:
      path_pretrain = os.path.join(dir_weights, 'pretrain/ImageNet-ResNet50-GN.pth')
      if not os.path.isfile(path_pretrain):
        url = 'https://github.com/ppwwyyxx/GroupNorm-reproduce/releases/download/v0.1/ImageNet-ResNet50-GN.pth'
        download_url(url, os.path.join(dir_weights, 'pretrain'))
      state_dict = torch.load(path_pretrain)['state_dict']
      state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
      state_dict.pop('fc.weight', None)
      state_dict.pop('fc.bias', None)
      print(f'{list(set(net.state_dict())-set(state_dict.keys()))} will be trained from scratch')
      net.load_state_dict(state_dict, strict=False)

  elif implementation == 'timm':
    # resnet50_gn: in1k, 23.7M parameters
    net = timm.create_model('resnet50_gn', pretrained=pretrained, num_classes=num_classes)

  else:
    raise NotImplementedError

  return net
