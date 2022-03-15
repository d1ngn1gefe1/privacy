import timm
from timm.models.layers import Mlp
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpAdapter(Mlp):
  def __init__(self, *args, **kargs):
    in_features = args[0]
    hidden_features, out_features = kargs['hidden_features'], kargs['out_features']
    out_features = out_features or in_features
    hidden_features = hidden_features or in_features
    super().__init__(*args, **kargs)

    self.adapter_fc1 = nn.Linear(out_features, out_features//4)
    self.adapter_fc2 = nn.Linear(out_features//4, out_features)

  def forward(self, x):
    # Original forward
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.fc2(x)
    x = self.drop2(x)
    # Adapter
    x_res = self.adapter_fc1(x)
    x_res = self.act(x_res)
    x_res = self.adapter_fc2(x_res)
    return x+x_res

  # Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py#L88
  @classmethod
  def convert_adapter(cls, module):
    mlp_module = Mlp
    res = module
    if isinstance(module, mlp_module):
      in_features = module.fc1.weight.shape[1]
      hidden_features = module.fc1.weight.shape[0]
      out_features = module.fc2.weight.shape[0]
      act_layer = module.act.__class__
      drop = module.drop1.p
      res = cls(in_features, hidden_features=hidden_features, out_features=out_features, act_layer=act_layer, drop=drop)

      # If pre-trained, copy weights
      res.fc1.weight.data = module.fc1.weight.data.clone().detach()
      res.fc1.bias.data = module.fc1.bias.data.clone().detach()
      res.fc2.weight.data = module.fc2.weight.data.clone().detach()
      res.fc2.bias.data = module.fc2.bias.data.clone().detach()
      # Adapter mode, freeze original parameters
      res.fc1.weight.requires_grad = False
      res.fc1.bias.requires_grad = False
      res.fc2.weight.requires_grad = False
      res.fc2.bias.requires_grad = False

    else:
      for name, child in module.named_children():
        new_child = cls.convert_adapter(child)
        if new_child is not child:
          res.add_module(name, new_child)

    return res

