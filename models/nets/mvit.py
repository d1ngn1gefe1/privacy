"""
Reference: https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/layers/positional_encoding.py
Examples of register_grad_sampler :
  - https://opacus.ai/tutorials/guide_to_grad_sampler
  - https://github.com/pytorch/opacus/blob/main/opacus/layers/dp_multihead_attention.py#L24
  - https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/dp_multihead_attention.py
"""

from opacus.grad_sample import register_grad_sampler
import os
from pytorchvideo.layers import SpatioTemporalClsPositionalEncoding
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
import torch
import torch.nn as nn
from torchvision.datasets.utils import download_url
from types import MethodType
from typing import Dict


@register_grad_sampler(SpatioTemporalClsPositionalEncoding)
def compute_spatio_temporal_cls_positional_encoding_sample(
    layer: SpatioTemporalClsPositionalEncoding, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  B, _, C = backprops.shape  # Bx(S*T)xC or Bx(S*T+1)xC
  ret = {}

  if layer.cls_embed_on:
    ret[layer.cls_token] = backprops[:, 0].unsqueeze(1).unsqueeze(1)  # Bx1x1xC

  if layer.sep_pos_embed:
    if layer.cls_embed_on:
      ret[layer.pos_embed_class] = backprops[:, 0].unsqueeze(1).unsqueeze(1)  # Bx1x1xC
      backprops_st = backprops[:, 1:]
    else:
      backprops_st = backprops

    backprops_st = backprops_st.reshape(B, layer.num_temporal_patch, layer.num_spatial_patch, C)
    ret[layer.pos_embed_spatial] = backprops_st.sum(dim=1).unsqueeze(1)  # Bx1xSxC
    ret[layer.pos_embed_temporal] = backprops_st.sum(dim=2).unsqueeze(1)  # Bx1xTxC

  else:
    ret[layer.pos_embed] = backprops.unsqueeze(1)  # Bx1xNxC

  return ret


def get_classifier(self):
  return self.head.proj


def get_mvit(num_classes, pretrained, dir_weights):
  net = create_multiscale_vision_transformers(
    spatial_size=224,
    temporal_size=16,
    cls_embed_on=True,
    sep_pos_embed=True,
    depth=16,
    norm='layernorm',
    input_channels=3,
    patch_embed_dim=96,
    conv_patch_embed_kernel=(3, 7, 7),
    conv_patch_embed_stride=(2, 4, 4),
    conv_patch_embed_padding=(1, 3, 3),
    enable_patch_embed_norm=False,
    use_2d_patch=False,
    # Attention block config,
    num_heads=1,
    mlp_ratio=4.0,
    qkv_bias=True,
    dropout_rate_block=0.0,
    droppath_rate_block=0.2,
    pooling_mode='conv',
    pool_first=False,
    embed_dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    atten_head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
    pool_q_stride_size=[[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]],
    pool_kv_stride_size=None,
    pool_kv_stride_adaptive=[1, 8, 8],
    pool_kvq_kernel=[3, 3, 3],
    # Head config,
    head=create_vit_basic_head,
    head_dropout_rate=0.5,
    head_activation=None,
    head_num_classes=num_classes
  )

  if pretrained:
    dir_pretrain = os.path.join(dir_weights, 'pretrain')
    fname_pretrain = 'MVIT_B_16x4.pyth'
    if not os.path.exists(os.path.join(dir_pretrain, fname_pretrain)):
      download_url(f'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/{fname_pretrain}', dir_pretrain)
    weights = torch.load(os.path.join(dir_pretrain, fname_pretrain))['model_state']
    weights.pop('head.proj.weight', None)
    weights.pop('head.proj.bias', None)
    print(f'{list(set(net.state_dict().keys())-set(weights.keys()))} will be trained from scratch')
    net.load_state_dict(weights, strict=False)
    net.get_classifier = MethodType(get_classifier, net)

  return net
