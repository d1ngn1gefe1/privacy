# Reference: https://github.com/pytorch/opacus/blob/main/examples/cifar10.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class OpacusNet(nn.Module):
  def __init__(self, num_classes, pretrained):
    super(OpacusNet, self).__init__()
    assert pretrained is False, 'Pre-trained weights not available'

    self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(128, num_classes, bias=True)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.avg_pool2d(x, kernel_size=2)
    x = F.relu(self.conv2(x))
    x = F.avg_pool2d(x, kernel_size=2)
    x = F.relu(self.conv3(x))
    x = F.avg_pool2d(x, kernel_size=2)
    x = F.relu(self.conv4(x))
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    x = self.fc1(x)

    return x

  def get_classifier(self):
    return self.fc1


def get_opacus_net(num_classes, pretrained):
  return OpacusNet(num_classes, pretrained)
