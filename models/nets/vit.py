"""
Reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Examples of register_grad_sampler :
  - https://opacus.ai/tutorials/guide_to_grad_sampler
  - https://github.com/pytorch/opacus/blob/main/opacus/layers/dp_multihead_attention.py#L24
  - https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/dp_multihead_attention.py
"""

from opacus.grad_sample import register_grad_sampler
import timm
from timm.models import vision_transformer
from timm.models.helpers import checkpoint_seq
import torch
import torch.nn as nn
from typing import Dict


class ParamEmbed(nn.Module):
  def __init__(self, cls_token, pos_embed):
    super().__init__()
    self.cls_token = cls_token
    self.pos_embed = pos_embed

  def forward(self, x):
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x+self.pos_embed
    return x


@register_grad_sampler(ParamEmbed)
def compute_param_embed_grad_sample(
    layer: ParamEmbed, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  ret = {
    layer.pos_embed: backprops.unsqueeze(1),
    layer.cls_token: backprops[:, 0].unsqueeze(1).unsqueeze(2)
  }
  return ret


class VisionTransformer(vision_transformer.VisionTransformer):
  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)
    self.param_embed = ParamEmbed(self.cls_token, self.pos_embed)

  def forward_features(self, x):
    x = self.patch_embed(x)
    x = self.param_embed(x)
    x = self.pos_drop(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
      x = checkpoint_seq(self.blocks, x)
    else:
      x = self.blocks(x)
    x = self.norm(x)
    return x

  @torch.jit.ignore
  def no_weight_decay(self):
    return {'param_embed.pos_embed', 'param_embed.cls_token'}

  def get_classifier(self):
    if self.param_embed.dist_token is None:
      return self.head
    else:
      return self.head, self.head_dist


def get_vit(num_classes, pretrained):
  vision_transformer.VisionTransformer = VisionTransformer
  # vit_tiny_patch16_224: in21k -> in1k, 21.7M parameters
  net = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=num_classes)
  delattr(net, 'cls_token')
  delattr(net, 'pos_embed')
  return net
