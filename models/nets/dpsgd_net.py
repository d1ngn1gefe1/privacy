# Reference: https://arxiv.org/pdf/1607.00133.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPSGDNet(nn.Module):
  def __init__(self, num_classes):
    super(DPSGDNet, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
    self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
    self.fc1 = nn.Linear(2304, 384, bias=True)
    self.fc2 = nn.Linear(384, num_classes, bias=True)

  def forward(self, x):
    # input size: 24x24x3
    x = F.relu(self.conv1(x))
    x = F.max_pool2d(x, 2)
    x = F.relu(self.conv2(x))
    x = F.max_pool2d(x, 2)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

  def get_classifier(self):
    return self.fc2

  @staticmethod
  def get_norms():
    return []


def get_dpsgd_net(cfg):
  assert cfg.mode == 'from_scratch' or cfg.weight is None, 'Pre-trained weights not available'
  net = DPSGDNet(cfg.num_classes)
  return net
