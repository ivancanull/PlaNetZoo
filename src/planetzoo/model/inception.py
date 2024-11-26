# https://github.com/A4Bio/SimVP/blob/master/modules.py#L52

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Callable

class GroupConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int, 
                 padding: int, 
                 groups: int, 
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 group_norm: bool = True,):
        super(GroupConv2d, self).__init__()
        layers = []
        
        if in_channels % groups != 0:
            print("in_channels % groups != 0, using groups = 1")
            groups = 1
        
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups))

        if group_norm:
            layers.append(nn.GroupNorm(groups, out_channels))
        
        if activation_layer is not None:
            layers.append(activation_layer())
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return y

class Inception(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 kernel_sizes: List[int] = [3,5,7,11],
                 groups: int = 8,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 group_norm: bool = True):        
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in kernel_sizes:
            layers.append(GroupConv2d(hidden_channels, out_channels, kernel_size=ker, stride=1, padding=ker//2, groups=groups, activation_layer=activation_layer, group_norm=group_norm))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y