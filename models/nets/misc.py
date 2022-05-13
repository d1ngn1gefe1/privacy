import torch.nn as nn


def get_norms(self):
  norms = []
  for module in self.modules():
    if isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
      norms.append(module)
  return norms
