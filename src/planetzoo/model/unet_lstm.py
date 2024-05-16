# Implementation of U-Net LSTM model.
# Papadomanolaki, Maria, et al. "Detecting urban changes with recurrent neural networks from multitemporal Sentinel-2 data." IGARSS 2019-2019 IEEE international geoscience and remote sensing symposium. IEEE, 2019.
# https://doi.org/10.1109/IGARSS.2019.8900330

# Reference Codes:
# https://github.com/mpapadomanolaki/UNetLSTM/blob/master/networks/networkL.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Callable

from .unet_parts import *
from .conv_lstm import ConvLSTM

class DownLSTM(nn.Module):
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
                 bias: bool = True,
                 batch_first=False, 
                 ):
        
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.batch_first = batch_first
        
        self.conv = DoubleConv(in_channels, 
                               out_channels, 
                               kernel_size=kernel_size, 
                               padding=padding,
                               norm_layer=norm_layer,
                               mid_channels=mid_channels,
                               activation_layer=activation_layer,
                               inplace=inplace,
                               bias=bias)
        
        self.pool = nn.MaxPool2d(2)
        self.conv_lstm = ConvLSTM(in_channels=out_channels,
                                  hidden_channels=[out_channels],
                                  batch_first=batch_first,
                                  return_all_layers=False)

    def forward(self, x):
        # reshape x from (b, t, c, h, w) => (b * t, c, h, w)
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, *x.size()[2:])
        x = self.pool(x)
        x = self.conv(x)
        # reshape x from (b * t, c, h, w) => (b, t, c, h, w)
        x = x.view(batch_size, seq_len, *x.size()[1:])
        layer_output_list, last_state_list = self.conv_lstm(x)

        return layer_output_list[-1], last_state_list[-1] # x, h
    

class UNetLSTM(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 out_channels: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, 
                 bilinear: bool = False,
                 batch_first=False, 
                 bias=True, 
                 return_all_layers=False):
        """
        
        Input Features:
            (batch_size, seq_len, in_channels, height, width) or (seq_len, batch_size, in_channels, height, width)
        
        Output Features:
            (batch_size, seq_len, hidden_channels[-1], height, width) or (seq_len, batch_size, hidden_channels[-1], height, width)

        """

        super().__init__()

        if len(hidden_channels) != 4:
            raise ValueError("Middle channel number must be 5!")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.conv1 = DoubleConv(self.in_channels, 
                                self.hidden_channels[0], 
                                kernel_size=5, 
                                padding=2,
                                norm_layer=norm_layer,
                                mid_channels=None,
                                activation_layer=activation_layer,
                                inplace=inplace,
                                bias=True)
        
        self.lstm1 = ConvLSTM(in_channels=self.hidden_channels[0],
                              hidden_channels=[self.hidden_channels[0]],
                              batch_first=batch_first,
                              return_all_layers=False)
        
        self.conv2 = DownLSTM(self.hidden_channels[0], 
                              self.hidden_channels[1], 
                              kernel_size=3,
                              padding=1,
                              norm_layer=norm_layer,
                              activation_layer=activation_layer,
                              inplace=inplace,
                              bias=True)
        
        self.conv3 = DownLSTM(self.hidden_channels[1],
                             self.hidden_channels[2],
                             kernel_size=3,
                             padding=1,
                             norm_layer=norm_layer,
                             activation_layer=activation_layer,
                             inplace=inplace,
                             bias=True)
        
        self.conv4 = DownLSTM(self.hidden_channels[2],
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

    def forward(self,
                x: torch.Tensor):
        
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.view(batch_size * seq_len, *x.size()[2:])
        conv1 = self.conv1(x)
        conv1 = conv1.view(batch_size, seq_len, *conv1.size()[1:])
        conv1, _ = self.lstm1(conv1)
        conv1 = conv1[-1]

        conv2, _ = self.conv2(conv1)
        conv3, _ = self.conv3(conv2)
        conv4, _ = self.conv4(conv3)

        up1 = self.up1(conv4.view(batch_size * seq_len, *conv4.size()[2:]), conv3.view(batch_size * seq_len, *conv3.size()[2:]))
        up2 = self.up2(up1, conv2.view(batch_size * seq_len, *conv2.size()[2:]))
        up3 = self.up3(up2, conv1.view(batch_size * seq_len, *conv1.size()[2:]))
        up4 = self.conv_last(up3)
        
        return up4.view(batch_size, seq_len, *up4.size()[1:])
    
if __name__ == "__main__":
    # test example
    device = torch.device("cuda:0")
    model = UNetLSTM(in_channels=1, hidden_channels=[64, 128, 256, 512], out_channels=1).to(device)
    model.initialize_weights(initial_func=torch.nn.init.kaiming_uniform_, 
                             func_args={"mode": "fan_in", "nonlinearity": "relu"})
    x = torch.randn((3, 30, 1, 56, 42)).to(device)
    y = model(x)
    print(y.shape)

