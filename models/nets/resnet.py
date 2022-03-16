import os
import torch
import torch.nn as nn
from torchvision import models
from types import MethodType


def get_classifier(self):
  return self.fc


def get_resnet(num_classes, pretrained, dir_weights):
  model = models.__dict__['resnet50'](pretrained=False, num_classes=num_classes,
                                      norm_layer=(lambda x: nn.GroupNorm(32, x)))
  if pretrained:
    # Download: https://github.com/ppwwyyxx/GroupNorm-reproduce/releases/download/v0.1/ImageNet-ResNet50-GN.pth
    state_dict = torch.load(os.path.join(dir_weights, 'pretrain/ImageNet-ResNet50-GN.pth'))['state_dict']
    state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    print(f'{list(set(model.state_dict())-set(state_dict.keys()))} will be trained from scratch')
    model.load_state_dict(state_dict, strict=False)

    model.get_classifier = MethodType(get_classifier, model)

  return model
