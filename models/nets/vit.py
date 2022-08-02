"""
Reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Examples of register_grad_sampler :
  - https://opacus.ai/tutorials/guide_to_grad_sampler
  - https://github.com/pytorch/opacus/blob/main/opacus/layers/dp_multihead_attention.py#L24
  - https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/dp_multihead_attention.py
"""

import clip
import clip.model
from opacus.grad_sample import register_grad_sampler
from opacus.validators import ModuleValidator
import os.path as osp
import timm
from timm.models import vision_transformer
from timm.models.helpers import checkpoint_seq
import torch
import torch.nn as nn
from types import MethodType
from typing import Dict

from .misc import get_norms


class EmbedTimm(nn.Module):
  def __init__(self, cls_token, pos_embed):
    super().__init__()
    self.cls_token = cls_token
    self.pos_embed = pos_embed

  def forward(self, x):
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x+self.pos_embed
    return x


@register_grad_sampler(EmbedTimm)
def compute_embed_timm_grad_sample(
    layer: EmbedTimm, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  ret = {
    layer.pos_embed: backprops.unsqueeze(1),
    layer.cls_token: backprops[:, 0].unsqueeze(1).unsqueeze(2)
  }
  return ret


class VisionTransformerTimm(vision_transformer.VisionTransformer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.embed_timm = EmbedTimm(self.cls_token, self.pos_embed)

  def forward_features(self, x):
    x = self.patch_embed(x)
    x = self.embed_timm(x)
    x = self.pos_drop(x)
    if self.grad_checkpointing and not torch.jit.is_scripting():
      x = checkpoint_seq(self.blocks, x)
    else:
      x = self.blocks(x)
    x = self.norm(x)
    return x

  @torch.jit.ignore
  def no_weight_decay(self):
    return {'embed_timm.pos_embed', 'embed_timm.cls_token'}


def delattrs_timm(net):
  delattr(net, 'cls_token')
  delattr(net, 'pos_embed')


class EmbedCLIP(nn.Module):
  def __init__(self, class_embedding, positional_embedding):
    super().__init__()
    self.class_embedding = class_embedding
    self.positional_embedding = positional_embedding

  def forward(self, x):
    x = torch.cat([self.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1), x], dim=1)
    x = x+self.positional_embedding
    return x


@register_grad_sampler(EmbedCLIP)
def compute_embed_clip_grad_sample(
    layer: EmbedCLIP, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  ret = {
    layer.positional_embedding: backprops,
    layer.class_embedding: backprops[:, 0]
  }
  return ret


def forward(self, x):
  x = x+self.attention(self.ln_1(x).permute(1, 0, 2)).permute(1, 0, 2)
  x = x+self.mlp(self.ln_2(x))
  return x


class Project(nn.Module):
  def __init__(self, proj):
    super().__init__()
    self.fc = nn.Linear(in_features=proj.shape[0], out_features=proj.shape[1], bias=False)
    self.fc.weight = nn.Parameter(torch.transpose(proj, 0, 1))

  def forward(self, x):
    return self.fc(x)


class VisionTransformerCLIP(nn.Module):
  def __init__(self, net, num_classes, extra_fc=False):
    super().__init__()

    self.visual = net.visual
    self.embed_clip = EmbedCLIP(self.visual.class_embedding, self.visual.positional_embedding)
    if extra_fc:
      self.proj = Project(self.visual.proj)
      self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
    else:
      self.fc = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    self.extra_fc = extra_fc

    delattr(self.visual, 'class_embedding')
    delattr(self.visual, 'positional_embedding')
    delattr(self.visual, 'proj')

  def forward(self, x: torch.Tensor):
    x = self.visual.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)
    x = self.embed_clip(x)
    x = self.visual.ln_pre(x)
    x = self.visual.transformer(x)
    x = self.visual.ln_post(x[:, 0, :])
    if self.extra_fc:
      x = self.proj(x)
    x = self.fc(x)
    return x

  def get_classifier(self):
    return self.fc


def get_vit(cfg):
  # vit_small_patch16_224: in21k -> in1k, 21.7M parameters
  if cfg.mode == 'from_scratch':
    print('Initializing randomly')
    vision_transformer.VisionTransformer = VisionTransformerTimm
    net = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=cfg.num_classes)
    delattrs_timm(net)

  elif cfg.weight == 'ckpt':
    print('Loading checkpoint')
    vision_transformer.VisionTransformer = VisionTransformerTimm
    net = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=cfg.num_classes)
    delattrs_timm(net)

    weight = torch.load(osp.join(cfg.dir_weights, cfg.rpath_ckpt))['state_dict']
    weight = {k.removeprefix('net.'):v for k, v in weight.items()}
    weight.pop('head.weight')
    weight.pop('head.bias')
    keys_missing, keys_unexpected = net.load_state_dict(weight, strict=False)
    assert len(keys_unexpected) == 0
    print(f'{keys_missing} will be trained from scratch')

  elif cfg.weight == 'pretrain':
    print('Loading ImageNet pre-trained weight (timm)')
    vision_transformer.VisionTransformer = VisionTransformerTimm
    net = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=cfg.num_classes)
    delattrs_timm(net)

  elif cfg.weight == 'pretrain_clip':
    print('Loading ImageNet pre-trained weight (CLIP)')
    clip.model.convert_weights = lambda model: None  # use fp32
    clip.model.LayerNorm = nn.LayerNorm  # no need to handle fp16
    clip.model.ResidualAttentionBlock.forward = forward
    net, _ = clip.load('ViT-B/16')
    net = VisionTransformerCLIP(net.train(), cfg.num_classes)
    net = ModuleValidator.fix(net)  # nn.MultiheadAttention -> opacus.layers.DPMultiheadAttention

  else:
    raise NotImplementedError

  net.get_norms = MethodType(get_norms, net)

  return net
