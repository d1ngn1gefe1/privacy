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
import torch
import torch.nn as nn
from typing import Dict


class ParamEmbed(nn.Module):
  def __init__(self, cls_token, dist_token, pos_embed):
    super().__init__()
    self.cls_token = cls_token
    self.dist_token = dist_token
    self.pos_embed = pos_embed

  def forward(self, x):
    cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    if self.dist_token is None:
      x = torch.cat((cls_token, x), dim=1)
    else:
      x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
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
  if layer.dist_token is not None:
    ret[layer.dist_token] = backprops[:, 1].unsqueeze(1).unsqueeze(2)
  return ret


class VisionTransformer(vision_transformer.VisionTransformer):
  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)
    self.param_embed = ParamEmbed(self.cls_token, self.dist_token, self.pos_embed)

  def forward_features(self, x):
    x = self.patch_embed(x)
    x = self.param_embed(x)
    x = self.pos_drop(x)
    x = self.blocks(x)
    x = self.norm(x)
    if self.param_embed.dist_token is None:
      return self.pre_logits(x[:, 0])
    else:
      return x[:, 0], x[:, 1]

  @torch.jit.ignore
  def no_weight_decay(self):
    return {'param_embed.pos_embed', 'param_embed.cls_token', 'param_embed.dist_token'}

  def get_classifier(self):
    if self.param_embed.dist_token is None:
      return self.head
    else:
      return self.head, self.head_dist


def get_vit(num_classes, pretrained):
  vision_transformer.VisionTransformer = VisionTransformer
  model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=pretrained, num_classes=num_classes)
  delattr(model, 'cls_token')
  delattr(model, 'dist_token')
  delattr(model, 'pos_embed')
  return model
