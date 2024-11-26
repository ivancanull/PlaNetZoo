# Inspired by:
# Gao, Zhangyang, et al. "Simvp: Simpler yet better video prediction." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

# A simple image to video prediction model
# Encoder - Translator - Decoder
# The Enocder consists of several Unet downsample blocks

import torch
import torch.nn as nn
import torch.nn.functional as F


from .unet_parts import *
from .inception import Inception
from typing import List, Optional, Callable

class Translator(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 nt: int,
                 kernel_sizes: List[int] = [3,5,7,11],
                 groups: int = 1,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 group_norm: bool = True):
        """
        
        Input Features:
            (batch_size, in_channels, height, width)
        Output Features:
            (batch_size, out_channels, height, width)


        """
        super().__init__()

        self.nt = nt
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        enc_layers = [Inception(in_channels, hidden_channels//2, hidden_channels, kernel_sizes=kernel_sizes, groups=groups, activation_layer=activation_layer, group_norm=group_norm)]
        for i in range(1, self.nt):
            enc_layers.append(Inception(hidden_channels, hidden_channels//2, hidden_channels, kernel_sizes=kernel_sizes, groups=groups, activation_layer=activation_layer, group_norm=group_norm))
        
        dec_layers = [Inception(hidden_channels, hidden_channels//2, hidden_channels, kernel_sizes=kernel_sizes, groups=groups, activation_layer=activation_layer, group_norm=group_norm)]
        for i in range(1, self.nt-1):
            dec_layers.append(Inception(2*hidden_channels, hidden_channels//2, hidden_channels, kernel_sizes=kernel_sizes, groups=groups, activation_layer=activation_layer, group_norm=group_norm))
        dec_layers.append(Inception(2*hidden_channels, hidden_channels//2, out_channels, kernel_sizes=kernel_sizes, groups=groups, activation_layer=activation_layer, group_norm=group_norm))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):

        # encoder
        skips = []
        z = x
        for i in range(self.nt):
            z = self.enc[i](z)
            if i < self.nt - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z)
        for i in range(1, self.nt):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

        return z

class UpGroupConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 groups: int,
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
            self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2, groups=groups)
    
        self.conv = DoubleConv(mid_channels * 2, out_channels, kernel_size, padding, None, norm_layer, activation_layer, inplace, bias, groups)

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
        
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class UpGroupConvNoSkip(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 groups: int,
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
        
        Output Features:
            (batch_size, out_channels, rows, cols)

        Operations:
            Upsample / ConvTranspose x1 => (batch_size, mid_channels, rows // 2, cols // 2)
            Padding x2 => (batch_size, mid_channels, rows, cols)
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
            self.up = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2, groups=groups)
    
        self.conv = DoubleConv(mid_channels, out_channels, kernel_size, padding, None, norm_layer, activation_layer, inplace, bias, groups)

    def forward(self, x1, x2_size):
        """
        x1: [n/2, n/2]
        """

        x1 = self.up(x1)
        # print(f"Up sample shape: {x1.shape}")
        # input is CHW
        diffY = x2_size[-2] - x1.size()[-2]
        diffX = x2_size[-1] - x1.size()[-1]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return self.conv(x1)

class SimI2VBase(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 out_channels: int,
                 seq_len: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, 
                 bilinear: bool = False,
                 batch_first=False, 
                 bias=True, 
        ):
        """
        
        Input Features:
            (batch_size, in_channels, rows, cols)

        Output Features:
            (batch_size, seq_len, out_channels, rows, cols)
        """ 

        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.seq_len = seq_len

        self.batch_first = batch_first
        self.bias = bias
    
    def _repeat_seq_len(self, tensor: torch.Tensor, squeeze=1):
        # if squeeze:
        # (B, C, H, W) => (B, T, C, H, W) (squeeze = 0)
        #              => (B * T, C, H, W) (squeeze = 1)
        #              => (B, T * C, H, W) (squeeze = 2)
        B, C = tensor.size()[0], tensor.size()[1]
        if squeeze == 0:
            return tensor.unsqueeze(1).repeat(1, self.seq_len, 1, 1, 1)
        elif squeeze == 1:
            return tensor.unsqueeze(1).repeat(1, self.seq_len, 1, 1, 1).view(B * self.seq_len, C, *tensor.size()[2:])
        elif squeeze == 2:
            return tensor.repeat(1, self.seq_len, 1, 1)
        else:
            raise ValueError()
    
    def initialize_weights(self, 
                           initial_func: Optional[Callable] = torch.nn.init.kaiming_normal_,
                           func_args: Optional[dict] = None,):

        layer_types = [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]
        for m in self.modules():
            if any(isinstance(m, layer) for layer in layer_types):
                initial_func(m.weight.data, **func_args)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

class SimI2V(nn.Module):
    """
    Encoder -> Translator -> Decoder
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 out_channels: int,
                 seq_len: int,
                 groups: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, 
                 bilinear: bool = False,
                 batch_first=False, 
                 bias=True, 
        ):
        """
        
        Input Features:
            (batch_size, in_channels, rows, cols)

        Output Features:
            (batch_size, seq_len, out_channels, rows, cols)
        """ 

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.seq_len = seq_len

        self.batch_first = batch_first
        self.bias = bias
        self.groups = groups

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

        self.translator = Translator(self.hidden_channels[3] * self.seq_len,
                                     self.hidden_channels[4] * self.seq_len,
                                     self.hidden_channels[3] * self.seq_len * self.groups,
                                     nt=3,
                                     groups=self.seq_len,
                                     activation_layer=activation_layer,
                                     group_norm=True
                                     )
        
        self.up1 = []
        self.up2 = []
        self.up3 = []

        for _ in range(self.groups):
            self.up1.append(UpGroupConv(self.hidden_channels[3], 
                                        self.hidden_channels[2],
                                        groups=1,
                                        kernel_size=3,
                                        padding=1,
                                        mid_channels=self.hidden_channels[2],
                                        norm_layer=norm_layer,
                                        activation_layer=activation_layer,
                                        inplace=inplace,
                                        bilinear=bilinear))
            
            self.up2.append(UpGroupConv(self.hidden_channels[2], 
                                        self.hidden_channels[1],
                                        groups=1,
                                        kernel_size=3,
                                        padding=1,
                                        mid_channels=self.hidden_channels[1],
                                        norm_layer=norm_layer,
                                        activation_layer=activation_layer,
                                        inplace=inplace,
                                        bilinear=bilinear))

            self.up3.append(UpGroupConv(self.hidden_channels[1], 
                                        self.hidden_channels[0],
                                        groups=1,
                                        kernel_size=3,
                                        padding=1,
                                        mid_channels=self.hidden_channels[0],
                                        norm_layer=norm_layer,
                                        activation_layer=activation_layer,
                                        inplace=inplace,
                                        bilinear=bilinear))
        self.up1 = nn.ModuleList(self.up1)
        self.up2 = nn.ModuleList(self.up2)
        self.up3 = nn.ModuleList(self.up3)

        # self.up1 = UpGroupConvNoSkip(self.hidden_channels[3] * self.groups, 
        #                             self.hidden_channels[2] * self.groups,
        #                             self.groups,
        #                             kernel_size=3,
        #                             padding=1,
        #                             norm_layer=norm_layer,
        #                             activation_layer=activation_layer,
        #                             inplace=inplace,
        #                             bilinear=bilinear)
        
        # self.up2 = UpGroupConvNoSkip(self.hidden_channels[2] * self.groups, 
        #                             self.hidden_channels[1] * self.groups,
        #                             self.groups,
        #                             kernel_size=3,
        #                             padding=1,
        #                             norm_layer=norm_layer,
        #                             activation_layer=activation_layer,
        #                             inplace=inplace,
        #                             bias=True,
        #                             bilinear=bilinear)
    
        # self.up3 = UpGroupConvNoSkip(self.hidden_channels[1] * self.groups, 
        #                             self.hidden_channels[0] * self.groups,
        #                             self.groups,
        #                             kernel_size=3,
        #                             padding=1,
        #                             norm_layer=norm_layer,
        #                             activation_layer=activation_layer,
        #                             inplace=inplace,
        #                             bias=True,
        #                             bilinear=bilinear)
    
        self.conv_last = nn.Conv2d(self.hidden_channels[0] * self.groups, 
                                    self.out_channels, 
                                    kernel_size=1,
                                    groups=self.groups)
    
    def _repeat_seq_len(self, tensor: torch.Tensor, seq_len, squeeze=1):
        # if squeeze:
        # (B, C, H, W) => (B, T, C, H, W) (squeeze = 0)
        #              => (B * T, C, H, W) (squeeze = 1)
        #              => (B, T * C, H, W) (squeeze = 2)
        B, C = tensor.size()[0], tensor.size()[1]
        if squeeze == 0:
            return tensor.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        elif squeeze == 1:
            return tensor.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(B * seq_len, C, *tensor.size()[2:])
        elif squeeze == 2:
            return tensor.repeat(1, seq_len, 1, 1)
        else:
            raise ValueError()
        
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
        B = x.size()[0]
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1) 
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        # repeat conv4 for seq_len times
        # assert self.hidden_channels[4] % self.hidden_channels[3] == 0
        
        conv4 = self._repeat_seq_len(conv4, seq_len=self.seq_len, squeeze=2) # (B, T * C, H, W)
        conv4 = self.translator(conv4)
        conv4 = conv4.view(B * self.seq_len, self.hidden_channels[3] * self.groups, *conv4.size()[-2:]) # (B * T, C, H, W)

        # repeat conv3, conv2, conv1 for seq_len times
        conv3 = self._repeat_seq_len(conv3, seq_len=self.seq_len, squeeze=1)
        conv2 = self._repeat_seq_len(conv2, seq_len=self.seq_len, squeeze=1)
        conv1 = self._repeat_seq_len(conv1, seq_len=self.seq_len, squeeze=1)

        up = []

        for i in range(self.groups):
            x = self.up1[i](conv4[:, i*self.hidden_channels[3]:(i+1)*self.hidden_channels[3], ...], conv3)
            x = self.up2[i](x, conv2)
            x = self.up3[i](x, conv1)
            up.append(x)
        
        x = torch.cat(up, dim=1)
        x = self.conv_last(x)

        x = x.view(B, self.seq_len, self.out_channels, *x.size()[-2:])

        return x

class SimI2VConnectDecNoSkip(nn.Module):
    """
    Encoder -> Translator -> Decoder
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 out_channels: int,
                 seq_len: int,
                 groups: List[int],
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, 
                 bilinear: bool = False,
                 batch_first=False, 
                 bias=True, 
        ):
        """
        
        Input Features:
            (batch_size, in_channels, rows, cols)

        Output Features:
            (batch_size, seq_len, out_channels, rows, cols)
        """ 

        super().__init__()   

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.groups = groups     


        self.batch_first = batch_first
        self.bias = bias
        
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

        # no translator
        # self.translator = Translator(self.hidden_channels[3] * self.seq_len,
        #                              self.hidden_channels[3] * self.seq_len,
        #                              self.hidden_channels[3] * self.seq_len,
        #                              nt=4,
        #                              groups=25,
        #                              activation_layer=activation_layer,
        #                              group_norm=True
        #                              )
        
        # 
        self.up1 = UpGroupConvNoSkip(self.hidden_channels[4], 
                                    self.hidden_channels[5],
                                    self.groups[0],
                                    kernel_size=3,
                                    padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    inplace=inplace,
                                    bilinear=bilinear)
    
        self.up2 = UpGroupConvNoSkip(self.hidden_channels[5], 
                                    self.hidden_channels[6],
                                    self.groups[1],
                                    kernel_size=3,
                                    padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    inplace=inplace,
                                    bias=True,
                                    bilinear=bilinear)
    
        self.up3 = UpGroupConvNoSkip(self.hidden_channels[6], 
                                    self.hidden_channels[7],
                                    self.groups[2],
                                    kernel_size=3,
                                    padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    inplace=inplace,
                                    bias=True,
                                    bilinear=bilinear)
    
        self.conv_last = nn.Conv2d(self.hidden_channels[7], 
                                    self.out_channels * self.seq_len, 
                                    kernel_size=1,
                                    groups=self.seq_len)

    def _repeat_seq_len(self, tensor: torch.Tensor, seq_len, squeeze=1):
        # if squeeze:
        # (B, C, H, W) => (B, T, C, H, W) (squeeze = 0)
        #              => (B * T, C, H, W) (squeeze = 1)
        #              => (B, T * C, H, W) (squeeze = 2)
        B, C = tensor.size()[0], tensor.size()[1]
        if squeeze == 0:
            return tensor.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        elif squeeze == 1:
            return tensor.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(B * seq_len, C, *tensor.size()[2:])
        elif squeeze == 2:
            return tensor.repeat(1, seq_len, 1, 1)
        else:
            raise ValueError()
        
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
        B = x.size()[0]
        x = self.conv1(x)
        conv1_size = x.size()[-2:]
        x = self.conv2(x) 
        conv2_size = x.size()[-2:]
        x = self.conv3(x)
        conv3_size = x.size()[-2:]
        x = self.conv4(x)

        # repeat conv4 for seq_len times
        assert self.hidden_channels[4] % self.hidden_channels[3] == 0
        
        x = self._repeat_seq_len(x, seq_len=self.hidden_channels[4] // self.hidden_channels[3], squeeze=2) # (B, T*C, H, W)

        # no translator

        x = self.up1(x, conv3_size)
        x = self.up2(x, conv2_size)
        x = self.up3(x, conv1_size)
        x = self.conv_last(x)
        x = x.view(B, self.seq_len, self.out_channels, *x.size()[-2:])

        return x

class SimI2VSepDecNoSkip(SimI2VBase):
    """
    Encoder -> Translator -> Decoder
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int],
                 out_channels: int,
                 seq_len: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                 inplace: Optional[bool] = None, 
                 bilinear: bool = False,
                 batch_first=False, 
                 bias=True, 
        ):
        """
        
        Input Features:
            (batch_size, in_channels, rows, cols)

        Output Features:
            (batch_size, seq_len, out_channels, rows, cols)
        """ 

        super().__init__(in_channels,
                         hidden_channels,
                         out_channels,
                         seq_len,
                         norm_layer,
                         activation_layer,
                         inplace, 
                         bilinear,
                         batch_first, 
                         bias)        
        
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

        # no translator
        # self.translator = Translator(self.hidden_channels[3] * self.seq_len,
        #                              self.hidden_channels[3] * self.seq_len,
        #                              self.hidden_channels[3] * self.seq_len,
        #                              nt=4,
        #                              groups=25,
        #                              activation_layer=activation_layer,
        #                              group_norm=True
        #                              )
        
        # 
        self.up1 = UpGroupConvNoSkip(self.hidden_channels[3] * self.seq_len, 
                                    self.hidden_channels[2] * self.seq_len,
                                    self.seq_len,
                                    kernel_size=3,
                                    padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    inplace=inplace,
                                    bilinear=bilinear)
    
        self.up2 = UpGroupConvNoSkip(self.hidden_channels[2] * self.seq_len, 
                                    self.hidden_channels[1] * self.seq_len,
                                    self.seq_len,
                                    kernel_size=3,
                                    padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    inplace=inplace,
                                    bias=True,
                                    bilinear=bilinear)
    
        self.up3 = UpGroupConvNoSkip(self.hidden_channels[1] * self.seq_len, 
                                    self.hidden_channels[0] * self.seq_len,
                                    self.seq_len,
                                    kernel_size=3,
                                    padding=1,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    inplace=inplace,
                                    bias=True,
                                    bilinear=bilinear)
    
        self.conv_last = nn.Conv2d(self.hidden_channels[0] * self.seq_len, 
                                    self.out_channels * self.seq_len, 
                                    kernel_size=1,
                                    groups=self.seq_len)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size()[0]
        x = self.conv1(x)
        conv1_size = x.size()[-2:]
        x = self.conv2(x) 
        conv2_size = x.size()[-2:]
        x = self.conv3(x)
        conv3_size = x.size()[-2:]
        x = self.conv4(x)

        # repeat conv4 for seq_len times
        x = self._repeat_seq_len(x, squeeze=2) # (B, T*C, H, W)

        # no translator

        x = self.up1(x, conv3_size)
        x = self.up2(x, conv2_size)
        x = self.up3(x, conv1_size)
        x = self.conv_last(x)
        x = x.view(B, self.seq_len, self.out_channels, *x.size()[-2:])

        return x

if __name__ == "__main__":

    from utils import *

    # test example
    device = torch.device("cuda:0")
    # model = SimI2V(in_channels=2, hidden_channels=[32, 64, 128, 256], out_channels=2, seq_len=100).to(device)

    model = SimI2V(in_channels=1, hidden_channels=[32, 64, 64, 64, 32], out_channels=6, seq_len=25, groups=3).to(device)
    count_parameters(model)
    x = torch.randn((4, 1, 56, 42)).to(device)
    y = model(x)
    print(y.shape)

    model = SimI2VSepDecNoSkip(in_channels=1, hidden_channels=[32, 64, 128, 256], out_channels=2, seq_len=25 * 3).to(device)
    count_parameters(model)
    x = torch.randn((4, 1, 56, 42)).to(device)
    y = model(x)
    print(y.shape)

    model = SimI2VConnectDecNoSkip(in_channels=1, hidden_channels=[32, 64, 128, 256, 256 * 5, 128 * 25, 64 * 25 * 3, 3 * 32 * 25], groups=[10, 25, 25 * 3], out_channels=6, seq_len=25).to(device)
    count_parameters(model)

    model.initialize_weights(initial_func=torch.nn.init.kaiming_uniform_, 
                             func_args={"mode": "fan_in", "nonlinearity": "relu"})

    x = torch.randn((4, 1, 56, 42)).to(device)
    y = model(x)
    print(y.shape)