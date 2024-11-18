"""
---
title: Train a ResNet on CIFAR 10
summary: >
  Train a ResNet on CIFAR 10
---

# Train a [ResNet](index.html) on CIFAR 10
"""
from typing import List, Optional

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.resnet import ResNetBase


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # Number fo blocks for each feature map size
    n_blocks: List[int] = [3, 3, 3]
    # Number of channels for each feature map size
    n_channels: List[int] = [16, 32, 64]
    # Bottleneck sizes
    bottlenecks: Optional[List[int]] = None
    # Kernel size of the initial convolution layer
    first_kernel_size: int = 3




def main():
    images = torch.randn((50, 4, 480, 640), dtype=torch.float32)  # 假设有 5 张 480x640 的 RGB 图像
    base = ResNetBase([3, 3, 3], [16, 32, 64], None, img_channels=4, first_kernel_size=5)
    model = nn.Sequential(base)
    output = model(images)
    print(output)

if __name__ == '__main__':
    main()