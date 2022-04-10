import os
from pytorchvideo.models.head import create_vit_basic_head
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers
import torch
from types import MethodType


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
    weights = torch.load(os.path.join(dir_weights, 'pretrain/MVIT_B_16x4.pyth'))['model_state']
    weights.pop('head.proj.weight', None)
    weights.pop('head.proj.bias', None)
    print(f'{list(set(net.state_dict().keys())-set(weights.keys()))} will be trained from scratch')
    net.load_state_dict(weights, strict=False)
    net.get_classifier = MethodType(get_classifier, net)

  return net
