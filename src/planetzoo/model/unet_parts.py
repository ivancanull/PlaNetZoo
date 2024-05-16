# Implementation of U-Net model parts.
# Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer International Publishing, 2015.
# https://doi.org/10.1007/978-3-319-24574-4_28

# Reference Codes:
# https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/unet.py
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Callable

__all__ = ["DoubleConv", "Down", "Up", "OutConv"]

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: Optional[int] = 3, 
                 padding: Optional[int] = 1, 
                 mid_channels: Optional[int] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None,
                 bias: bool = True,
                 ):
        
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        layers = []

        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        if norm_layer is not None:
            layers.append(norm_layer(mid_channels))
        layers.append(activation_layer(inplace=inplace))

        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        layers.append(activation_layer(inplace=inplace))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: Optional[int] = 3, 
                 padding: Optional[int] = 1, 
                 mid_channels: Optional[int] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None,
                 bias: bool = True):
        
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size, padding, mid_channels, norm_layer, activation_layer, inplace, bias)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: Optional[int] = 3, 
                 padding: Optional[int] = 1, 
                 mid_channels: Optional[int] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = nn.BatchNorm2d,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None,
                 bias: bool = True,
                 bilinear: bool = False):
        """
        
        Input Features:
            x1 - (batch_size, in_channels, rows // 2, cols // 2)
            x2 - (batch_size, mid_channels, rows, cols)
        
        Output Features:
            (batch_size, out_channels, rows, cols)

        Operations:
            Upsample / ConvTranspose x1 => (batch_size, mid_channels, rows // 2, cols // 2)
            Padding x2 => (batch_size, mid_channels, rows, cols)
            Concatenate x1, x2 => (batch_size, mid_channels * 2, rows, cols)
            DoubleConv => (batch_size, out_channels, rows, cols)
        
        Default:
            mid_channels = out_channels // 2

        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels // 2

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
    
        self.conv = DoubleConv(mid_channels * 2, out_channels, kernel_size, padding, None, norm_layer, activation_layer, inplace, bias)

    def forward(self, x1, x2):
        """
        x1: [n/2, n/2]
        x2: [n, n], skip connection
        """

        x1 = self.up(x1)
        # print(f"Up sample shape: {x1.shape}")
        # input is CHW
        diffY = x2.size()[-2] - x1.size()[-2]
        diffX = x2.size()[-1] - x1.size()[-1]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)