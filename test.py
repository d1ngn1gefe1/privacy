from opacus.validators.module_validator import ModuleValidator
import timm
# from opacus.layers import SequenceBias
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from opacus.grad_sample import register_grad_sampler
from typing import Dict


class SequenceBias(nn.Module):
  def __init__(self, embed_dim: int = 10, batch_first: bool = False):
    super(SequenceBias, self).__init__()
    self.batch_first = batch_first
    self.bias = Parameter(torch.empty(embed_dim))
    self._reset_parameters()

  def _reset_parameters(self):
    nn.init.normal_(self.bias)

  def forward(self, x):
    if self.batch_first:
      bsz, _, _ = x.shape
      return torch.cat([x, self.bias.repeat(bsz, 1, 1)], 1)
    else:
      _, bsz, _ = x.shape
      return torch.cat([x, self.bias.repeat(1, bsz, 1)])


@register_grad_sampler(SequenceBias)
def compute_sequence_bias_grad_sample(
    layer: SequenceBias, activations: torch.Tensor, backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
  """
  Computes per sample gradients for ``SequenceBias`` layer
  Args:
      layer: Layer
      activations: Activations
      backprops: Backpropagations
  """
  return {layer.bias: backprops[:, -1]}


def main():
  # model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True, num_classes=10)
  # ModuleValidator.validate(model.patch_embed, strict=True)
  # ModuleValidator.validate(model.pos_drop, strict=True)
  # ModuleValidator.validate(model.blocks, strict=True)
  # ModuleValidator.validate(model.pre_logits, strict=True)
  # ModuleValidator.validate(model.head, strict=True)

  model = SequenceBias(10)
  ModuleValidator.validate(model, strict=True)


if __name__ == '__main__':
  main()
