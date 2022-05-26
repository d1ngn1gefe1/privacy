from clip.model import ResidualAttentionBlock
from opacus.validators import ModuleValidator
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpClipAdapter(ResidualAttentionBlock):
  def __init__(self, *args, **kargs):
    self.d_model = args[0]
    super().__init__(*args, **kargs)

    self.attn = ModuleValidator.fix(self.attn)  # nn.MultiheadAttention -> opacus.layers.DPMultiheadAttention
    self.act = nn.GELU()
    self.adapter_fc1 = nn.Linear(self.d_model, self.d_model//4)
    self.adapter_fc2 = nn.Linear(self.d_model//4, self.d_model)

  def forward(self, x):
    # Original forward
    x = x + self.attention(self.ln_1(x))
    identity = x
    x = self.mlp(self.ln_2(x))
    # Adapter
    x_res = self.adapter_fc1(x)
    x_res = self.act(x_res)
    x_res = self.adapter_fc2(x_res)
    return x+x_res+identity

  # Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py#L88
  @classmethod
  def convert_adapter(cls, module):
    mlp_module = ResidualAttentionBlock
    res = module
    if isinstance(module, mlp_module):
      d_model = module.mlp.c_fc.weight.shape[1]
      n_head = module.attn.num_heads
      res = cls(d_model, n_head, attn_mask=None)

      # TODO: Simplify this
      # If pre-trained, copy weights
      res.attn.qlinear.weight.data = module.attn.qlinear.weight.data.clone().detach()
      res.attn.qlinear.bias.data = module.attn.qlinear.bias.data.clone().detach()
      res.attn.klinear.weight.data = module.attn.klinear.weight.data.clone().detach()
      res.attn.klinear.bias.data = module.attn.klinear.bias.data.clone().detach()
      res.attn.vlinear.weight.data = module.attn.vlinear.weight.data.clone().detach()
      res.attn.vlinear.bias.data = module.attn.vlinear.bias.data.clone().detach()
      res.attn.out_proj.weight.data = module.attn.out_proj.weight.data.clone().detach()
      res.attn.out_proj.bias.data = module.attn.out_proj.bias.data.clone().detach()
      res.ln_1.weight.data = module.ln_1.weight.data.clone().detach()
      res.ln_1.bias.data = module.ln_1.bias.data.clone().detach()
      res.mlp.c_fc.weight.data = module.mlp.c_fc.weight.data.clone().detach()
      res.mlp.c_fc.bias.data = module.mlp.c_fc.bias.data.clone().detach()
      res.mlp.c_proj.weight.data = module.mlp.c_proj.weight.data.clone().detach()
      res.mlp.c_proj.bias.data = module.mlp.c_proj.bias.data.clone().detach()
      res.ln_2.weight.data = module.ln_2.weight.data.clone().detach()
      res.ln_2.bias.data = module.ln_2.bias.data.clone().detach()

      # Adapter mode, freeze original parameters
      for name, param in res.named_parameters():
        if 'adapter' not in name:
          param.requires_grad = False

    else:
      for name, child in module.named_children():
        new_child = cls.convert_adapter(child)
        if new_child is not child:
          res.add_module(name, new_child)

    return res

