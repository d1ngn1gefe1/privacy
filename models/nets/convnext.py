from functools import partial
from opacus.utils.tensor_utils import sum_over_all_but_batch_and_last_n
from opacus.grad_sample import register_grad_sampler
import timm
from timm.models import convnext
from timm.models.convnext import LayerNorm2d
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


@register_grad_sampler(LayerNorm2d)
def compute_layer_norm_grad_sample(
    layer: LayerNorm2d,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
  # fixme
  N, C, H, W = activations.shape
  temp = activations.permute(0, 2, 3, 1).reshape(-1, C)
  x_norm = F.layer_norm(temp, layer.normalized_shape, eps=layer.eps)
  x_norm = torch.sum(x_norm.reshape(N, H, W, C), dim=(1, 2))
  return {
    layer.weight: x_norm * torch.sum(backprops, dim=(2, 3)),
    layer.bias: torch.sum(backprops, dim=(2, 3))
  }


class GammaEmbed(nn.Module):
  def __init__(self, dim, ls_init_value):
    super().__init__()
    self.gamma = nn.Parameter(ls_init_value*torch.ones(dim))

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


class ConvNeXtBlock(nn.Module):
  def __init__(self, dim, drop_path=0., ls_init_value=1e-6, conv_mlp=False, mlp_ratio=4, norm_layer=None):
    super().__init__()
    if not norm_layer:
      norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
    mlp_layer = ConvMlp if conv_mlp else Mlp
    self.use_conv_mlp = conv_mlp
    self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
    self.norm = norm_layer(dim)
    self.mlp = mlp_layer(dim, int(mlp_ratio*dim), act_layer=nn.GELU)
    self.gamma_embed = GammaEmbed(dim, ls_init_value) if ls_init_value > 0 else None
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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


def get_convnext(num_classes, pretrained):
  convnext.ConvNeXtBlock = ConvNeXtBlock
  net = timm.create_model('convnext_base_in22ft1k', pretrained=pretrained, num_classes=num_classes)
  return net
