from functools import partial
from opacus.grad_sample import register_grad_sampler
import os.path as osp
import timm
from timm.models.layers.norm import GroupNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets.utils import download_url
from types import MethodType
from typing import Dict

from .misc import get_norms


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


def get_resnet(cfg, implementation='ppwwyyxx'):
  assert implementation in ['ppwwyyxx', 'timm']

  if implementation == 'ppwwyyxx':
    net = models.__dict__[cfg.net](pretrained=False, num_classes=cfg.num_classes,
                                   norm_layer=partial(nn.GroupNorm, 32 if cfg.net == 'resnet50' else 8))

    if cfg.mode == 'from_scratch':
      print('Initializing randomly')

    elif cfg.weight == 'ckpt':
      print('Loading checkpoint')
      weight = torch.load(osp.join(cfg.dir_weights, cfg.rpath_ckpt))['state_dict']
      weight = {k.removeprefix('net.'):v for k, v in weight.items()}
      weight.pop('fc.weight')
      weight.pop('fc.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    else:
      assert cfg.weight == 'pretrain' and cfg.net == 'resnet50'
      print('Loading ImageNet pre-trained weight')
      path_pretrain = osp.join(cfg.dir_weights, 'pretrain/ImageNet-ResNet50-GN.pth')
      if not osp.isfile(path_pretrain):
        url = 'https://github.com/ppwwyyxx/GroupNorm-reproduce/releases/download/v0.1/ImageNet-ResNet50-GN.pth'
        download_url(url, osp.join(cfg.dir_weights, 'pretrain'))

      weight = torch.load(path_pretrain)['state_dict']
      weight = {k.removeprefix('module.'):v for k, v in weight.items()}
      weight.pop('fc.weight')
      weight.pop('fc.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    net.get_classifier = MethodType(get_classifier, net)
    net.get_norms = MethodType(get_norms, net)

  elif implementation == 'timm':
    # resnet50_gn: in1k, 23.7M parameters
    if cfg.mode == 'full_tuning':
      print('Initializing randomly')
      net = timm.create_model('resnet50_gn', pretrained=False, num_classes=cfg.num_classes)

    elif cfg.weight == 'ckpt':
      print('Loading checkpoint')
      net = timm.create_model('resnet50_gn', pretrained=False, num_classes=cfg.num_classes)

      weight = torch.load(osp.join(cfg.dir_weights, cfg.rpath_ckpt))['state_dict']
      weight = {k.removeprefix('net.'): v for k, v in weight.items()}
      weight.pop('fc.weight')
      weight.pop('fc.bias')
      keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
      assert len(keys_unexpected) == 0
      print(f'{keys_missing} will be trained from scratch')

    else:
      assert cfg.weight == 'pretrain'
      print('Loading ImageNet pre-trained weight')
      net = timm.create_model('resnet50_gn', pretrained=True, num_classes=cfg.num_classes)

    net.get_norms = MethodType(get_norms, net)

  else:
    raise NotImplementedError

  return net
