# Reference: https://arxiv.org/pdf/1607.00133.pdf

import torch.nn as nn


def DPSGDNet(num_classes):
  """Input size 24x24x3"""
  return nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),    # 12x12x64
    nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),    # 6x6x64
    nn.Flatten(),
    nn.Linear(2304, 384, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(384, num_classes, bias=True)
  )

