"""
Reference:
 - https://github.com/facebookresearch/ConvNeXt
 - https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/layer_norm.py#L27
 - https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/normalization.py#L188
"""

from opacus.grad_sample import register_grad_sampler
import os.path as osp
import timm
from timm.models import convnext
from timm.models.convnext import LayerNorm2d, _is_contiguous
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from typing import Dict

from .misc import get_norms


@register_grad_sampler(LayerNorm2d)
def compute_layer_norm_timm_grad_sample(
    layer: LayerNorm2d,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
  if _is_contiguous(activations):
    x = F.layer_norm(activations.permute(0, 2, 3, 1), layer.normalized_shape, layer.weight, layer.bias,
                     layer.eps).permute(0, 3, 1, 2)
  else:
    s, u = torch.var_mean(activations, dim=1, unbiased=False, keepdim=True)
    x = (activations-u)*torch.rsqrt(s+layer.eps)

  ret = {
    layer.weight: torch.sum(x*backprops, dim=(2, 3)),
    layer.bias: torch.sum(backprops, dim=(2, 3))
  }

  return ret


class GammaEmbed(nn.Module):
  def __init__(self, gamma):
    super().__init__()
    self.gamma = gamma

  def forward(self, x):
    x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    return x


@register_grad_sampler(GammaEmbed)
def compute_param_embed_grad_sample(
    layer: GammaEmbed, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  ret = {
    layer.gamma: torch.einsum('nijk,nijk->ni', backprops, activations)
  }
  return ret


class ConvNeXtBlock(convnext.ConvNeXtBlock):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.gamma is not None:
      self.gamma_embed = GammaEmbed(self.gamma)

  def forward(self, x):
    shortcut = x
    x = self.conv_dw(x)
    if self.use_conv_mlp:
      x = self.norm(x)
      x = self.mlp(x)
    else:
      x = x.permute(0, 2, 3, 1)
      x = self.norm(x)
      x = self.mlp(x)
      x = x.permute(0, 3, 1, 2)
    if self.gamma_embed is not None:
      x = self.gamma_embed(x)
    x = self.drop_path(x)+shortcut
    return x


def delattrs(net):
  for stage in net.stages:
    for block in stage.blocks:
      delattr(block, 'gamma')


def get_convnext(cfg):
  convnext.ConvNeXtBlock = ConvNeXtBlock

  # convnext_tiny_in22ft1k: in21k -> in1k, 29.5M parameters
  if cfg.mode == 'from_scratch':
    print('Initializing randomly')
    net = timm.create_model('convnext_tiny_in22ft1k', pretrained=False, num_classes=cfg.num_classes)
    delattrs(net)

  elif cfg.weight == 'ckpt':
    print('Loading checkpoint')
    net = timm.create_model('convnext_tiny_in22ft1k', pretrained=False, num_classes=cfg.num_classes)
    delattrs(net)

    weight = torch.load(osp.join(cfg.dir_weights, cfg.rpath_ckpt))['state_dict']
    weight = {k.removeprefix('net.'): v for k, v in weight.items()}
    weight.pop('head.fc.weight')
    weight.pop('head.fc.bias')
    keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
    assert len(keys_unexpected) == 0
    print(f'{keys_missing} will be trained from scratch')

  else:
    print('Loading ImageNet pre-trained weight')
    net = timm.create_model('convnext_tiny_in22ft1k', pretrained=True, num_classes=cfg.num_classes)
    delattrs(net)

  net.get_norms = MethodType(get_norms, net)

  return net
