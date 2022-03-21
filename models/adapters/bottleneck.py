import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck


class BottleneckAdapter(Bottleneck):
  expansion: int = 4
  def __init__(self, *args, **kargs):
    super().__init__(*args, **kargs)
    out_features = self.bn3.weight.shape[0]//self.expansion

    self.adapter_conv1 = nn.Conv2d(out_features, out_features//32, kernel_size=1, stride=1, bias=True)
    self.adapter_conv2 = nn.Conv2d(out_features//32, out_features, kernel_size=1, stride=1, bias=True)
    self.act = nn.GELU()

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    # Adapter
    res = self.adapter_conv1(out)
    res = self.act(res)
    res = self.adapter_conv2(res)
    out = out+res

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out

  # Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py#L88
  @classmethod
  def convert_adapter(cls, module):
    mlp_module = Bottleneck
    res = module
    if isinstance(module, mlp_module):
      inplanes = module.conv1.weight.shape[1]  # assume groups==1
      width = module.conv1.weight.shape[0]
      planes = module.conv3.weight.shape[0]
      norm_layer = lambda x: nn.GroupNorm(32, x)
      res = cls(inplanes, planes, stride=module.stride, downsample=module.downsample, norm_layer=norm_layer)

      # If pre-trained, copy weights
      res.conv1.weight.data = module.conv1.weight.data.clone().detach()
      res.bn1.weight.data = module.bn1.weight.data.clone().detach()
      res.bn1.bias.data = module.bn1.bias.data.clone().detach()
      res.conv2.weight.data = module.conv2.weight.data.clone().detach()
      res.bn2.weight.data = module.bn2.weight.data.clone().detach()
      res.bn2.bias.data = module.bn2.bias.data.clone().detach()
      res.conv3.weight.data = module.conv3.weight.data.clone().detach()
      res.bn3.weight.data = module.bn3.weight.data.clone().detach()
      res.bn3.bias.data = module.bn3.bias.data.clone().detach()

      # Adapter mode, freeze original parameters
      res.conv1.weight.requires_grad = False
      res.bn1.weight.requires_grad = False
      res.bn1.bias.requires_grad = False
      res.conv2.weight.requires_grad = False
      res.bn2.weight.requires_grad = False
      res.bn2.bias.requires_grad = False
      res.conv3.weight.requires_grad = False
      res.bn3.weight.requires_grad = False
      res.bn3.bias.requires_grad = False

    else:
      for name, child in module.named_children():
        new_child = cls.convert_adapter(child)
        if new_child is not child:
          res.add_module(name, new_child)

    return res

