# Implementation of U-Net model.
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
from .unet_parts import *

__all__ = ["UNet"]

class UNet(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 out_channels: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, 
                 bilinear: bool = False,
        ):
        """
        Initialize a U-Net model.

        Args:
            in_channels (int): Number of input channels.
            hidden_channels (List[int]): List of hidden layer dimensions.

        Input Features:
            (batch_size, in_channels, rows, cols)
        Output Features:
            (batch_size, out_channels, rows, cols)
        """
        super().__init__()

        if len(hidden_channels) != 4:
            raise ValueError("Middle channel number must be 4!")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.conv1 = DoubleConv(self.in_channels, 
                                self.hidden_channels[0], 
                                kernel_size=5, 
                                padding=2,
                                norm_layer=norm_layer,
                                mid_channels=None,
                                activation_layer=activation_layer,
                                inplace=inplace,
                                bias=True)
        
        self.conv2 = Down(self.hidden_channels[0], 
                          self.hidden_channels[1], 
                          kernel_size=3,
                          padding=1,
                          norm_layer=norm_layer,
                          activation_layer=activation_layer,
                          inplace=inplace,
                          bias=True)
        
        self.conv3 = Down(self.hidden_channels[1], 
                          self.hidden_channels[2],
                          kernel_size=3,
                          padding=1,
                          norm_layer=norm_layer,
                          activation_layer=activation_layer,
                          inplace=inplace,
                          bias=True)
        
        self.conv4 = Down(self.hidden_channels[2], 
                          self.hidden_channels[3],
                          kernel_size=3,
                          padding=1,
                          norm_layer=norm_layer,
                          activation_layer=activation_layer,
                          inplace=inplace,
                          bias=True)

        self.up1 = Up(self.hidden_channels[3], 
                      self.hidden_channels[2] * 2,
                      kernel_size=3,
                      padding=1,
                      norm_layer=norm_layer,
                      activation_layer=activation_layer,
                      inplace=inplace,
                      bilinear=bilinear)
        
        self.up2 = Up(self.hidden_channels[2] * 2, 
                      self.hidden_channels[1] * 2,
                      kernel_size=3,
                      padding=1,
                      norm_layer=norm_layer,
                      activation_layer=activation_layer,
                      inplace=inplace,
                      bias=True,
                      bilinear=bilinear)
        
        self.up3 = Up(self.hidden_channels[1] * 2, 
                      self.hidden_channels[0] * 2,
                      kernel_size=3,
                      padding=1,
                      norm_layer=norm_layer,
                      activation_layer=activation_layer,
                      inplace=inplace,
                      bias=True,
                      bilinear=bilinear)
        
        self.conv_last = nn.Conv2d(self.hidden_channels[0] * 2, 
                                   self.out_channels, 
                                   kernel_size=1)
    
    def initialize_weights(self, 
                           initial_func: Optional[Callable] = torch.nn.init.kaiming_normal_,
                           func_args: Optional[dict] = None,):

        layer_types = [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]
        for m in self.modules():
            if any(isinstance(m, layer) for layer in layer_types):
                initial_func(m.weight.data, **func_args)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1) 
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        
        up1 = self.up1(conv4, conv3)
        up2 = self.up2(up1, conv2)
        up3 = self.up3(up2, conv1)
        up4 = self.conv_last(up3)

        return up4
    
if __name__ == "__main__":
    # test example
    device = torch.device("cuda:0")
    model = UNet(in_channels=1, hidden_channels=[64, 128, 256, 512], out_channels=1).to(device)
    model.initialize_weights(initial_func=torch.nn.init.kaiming_uniform_, 
                             func_args={"mode": "fan_in", "nonlinearity": "relu"})
    x = torch.randn((1, 1, 56, 42)).to(device)
    y = model(x)
    print(y.shape)