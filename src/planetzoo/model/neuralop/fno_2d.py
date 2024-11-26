# Implement the Fourier Neural Operator 2D model.
# Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).
# https://arxiv.org/abs/2010.08895

# Reference Codes:
# https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/fno.py

from re import X
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pyutils.activation import Swish
# from timm.models.layers import DropPath
from torch import nn
from torch.functional import Tensor

from ..unet import *
from ...layers.fno_conv2d import FNOConv2d

__all__ = ["FNO2dBlock", "FNO2D", "FNO2DRNN", "FNO2DGRU"]

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        # device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            # self.act_func = Swish()
            raise NotImplementedError
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.act_func is not None:
            x = self.act_func(x)
        return x
    
class FNO2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int],
        kernel_size: int = 1,
        padding: int = 0,
        act_func: Optional[str] = "GELU",
        # drop_path_rate: float = 0.0,
        # device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        # self.drop_path_rate = drop_path_rate
        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.f_conv = FNOConv2d(in_channels, out_channels, n_modes)
        self.norm = nn.BatchNorm2d(out_channels)
        # self.norm.weight.data.zero_()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        if act_func is None:
            self.act_func = None
        elif act_func.lower() == "swish":
            # self.act_func = Swish()
            raise NotImplementedError
        else:
            self.act_func = getattr(nn, act_func)()

    def forward(self, x: Tensor) -> Tensor:
        # x = self.norm(self.conv(x) + self.drop_path(self.f_conv(x)))
        x = self.norm(self.conv(x) + self.f_conv(x))

        if self.act_func is not None:
            x = self.act_func(x)
        return x

class FNO2D(nn.Module):
    """
    The structures of Fourier Neural Operator, reference to 
    https://github.com/JeremieMelo/NeurOLight/blob/main/core/models/fno_cnn.py
    http://arxiv.org/abs/2209.10098

    The network consists of a stem and head consisted of conv layers, and a feature
    network with n FNO2dBlock.
    The network takes (B, Cin, H, L) as input
    and outputs is (B, Cout, H, L).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        dim: int = 16,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        act_func: Optional[str] = "GELU",
    ) -> None:
        """
        :param in_channels: input channel num
        :param out_channels: output channel num
        :param dim: channel number after stem
        :param kernel_list: kernel numbers of each fourier operator block
        :param kernel_size_list: kernel sizes of each fourier operator conv pass
        :param mode_list: the modes to keep in each fourier operator block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func

        self.build_layer()
    
    def build_layer(self):

        # Define stem net
        self.stem = nn.Conv2d(
            self.in_channels,
            self.dim,
            1,
            padding=0,
        )

        # Define features net
        kernel_list = [self.dim] + self.kernel_list
        features = [
            FNO2dBlock(
                in_channels,
                out_channels,
                n_modes,
                kernel_size,
                padding,
                act_func=self.act_func,
            )
            for in_channels, out_channels, n_modes, kernel_size, padding in zip(
                kernel_list[:-1],
                kernel_list[1:],
                self.mode_list,
                self.kernel_size_list,
                self.padding_list,
            )
        ]
        self.features = nn.Sequential(*features)

        # Define head net
        hidden_list = [self.kernel_list[-1]] + self.hidden_list
        head = [
            nn.Sequential(
                ConvBlock(inc, outc, kernel_size=1, padding=0, act_func=self.act_func,),
                # nn.Dropout2d(self.dropout_rate),
            )
            for inc, outc in zip(hidden_list[:-1], hidden_list[1:])
        ]
        head += [
            ConvBlock(
                hidden_list[-1],
                self.out_channels,
                kernel_size=1,
                padding=0,
                act_func=None,
            )
        ]
        self.head = nn.Sequential(*head)
    
    def initialize_weights(self, 
                           initial_func: Optional[Callable] = torch.nn.init.kaiming_normal_,
                           func_args: Optional[dict] = None,):

        layer_types = [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]
        for m in self.modules():
            if any(isinstance(m, layer) for layer in layer_types):
                initial_func(m.weight.data, **func_args)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data) 
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x

class FNO2DRNN(nn.Module):
    """
    FNO2D + RNN Structure
    
    The network takes (B, Cin, H, L) as input
    and outputs is (B, T, Cout, H, L).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        dim: int = 16,
        seq_len: int = 25,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        act_func: Optional[str] = "GELU",
    ) -> None:
        """
        :param in_channels: input channel num
        :param out_channels: output channel num
        :param dim: channel number after stem
        :param kernel_list: kernel numbers of each fourier operator block
        :param kernel_size_list: kernel sizes of each fourier operator conv pass
        :param mode_list: the modes to keep in each fourier operator block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.seq_len = seq_len
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func

        self.build_layer()
    
    def build_layer(self):
        self.unet = UNet(
            self.in_channels,
            [32, 64, 128, 256],
            self.out_channels,)
        
    def initialize_weights(self, 
                           initial_func: Optional[Callable] = torch.nn.init.kaiming_normal_,
                           func_args: Optional[dict] = None,):

        self.unet.initialize_weights(initial_func, func_args)
        self.fno.initialize_weights(initial_func, func_args)

    def forward(self, x: Tensor) -> Tensor:
        
        y = self.unet(x)
        output = [y]
        for t in range(self.seq_len - 1):
            y = self.fno(y)
            output.append(y)
            
        return torch.stack(output, dim=1)

class FNO2DGRU(nn.Module):
    """
    FNO2D + GRU Structure
    
    The network takes (B, Cin, H, L) as input
    and outputs is (B, T, Cout, H, L).
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        gru_hidden_channels: int = 16,
        dim: int = 16,
        seq_len: int = 25,
        kernel_list: List[int] = [16, 16, 16, 16],
        kernel_size_list: List[int] = [1, 1, 1, 1],
        padding_list: List[int] = [0, 0, 0, 0],
        hidden_list: List[int] = [128],
        mode_list: List[Tuple[int]] = [(20, 20), (20, 20), (20, 20), (20, 20)],
        act_func: Optional[str] = "GELU",
        unet: bool = False
    ) -> None:
        """
        :param in_channels: input channel num
        :param out_channels: output channel num
        :param dim: channel number after stem
        :param kernel_list: kernel numbers of each fourier operator block
        :param kernel_size_list: kernel sizes of each fourier operator conv pass
        :param mode_list: the modes to keep in each fourier operator block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru_hidden_channels = gru_hidden_channels
        self.dim = dim
        self.seq_len = seq_len
        self.kernel_list = kernel_list
        self.kernel_size_list = kernel_size_list
        self.padding_list = padding_list
        self.hidden_list = hidden_list
        self.mode_list = mode_list
        self.act_func = act_func
        self.unet = unet

        self.build_layer()
    
    def build_layer(self):
        if self.unet:
            self.unet = FNO2D(
                in_channels=self.in_channels, out_channels=self.out_channels, dim=64, kernel_list=[64, 64, 64, 64], kernel_size_list=[1, 1, 1, 1], padding_list=[0, 0, 0, 0], hidden_list=[128], mode_list=[(16, 16), (16, 16), (16, 16), (16, 16)], act_func=self.act_func)
        
        self.fno = FNO2D(
            self.out_channels if self.unet else self.in_channels,
            self.out_channels,
            self.dim,
            self.kernel_list,
            self.kernel_size_list,
            self.padding_list,
            self.hidden_list,
            self.mode_list,
            self.act_func,
        )

        self.xr = nn.Sequential(nn.Conv2d(self.out_channels, self.gru_hidden_channels, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.gru_hidden_channels, self.out_channels, 3, padding=1))
                
        self.xz = nn.Sequential(nn.Conv2d(self.out_channels, self.gru_hidden_channels, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.gru_hidden_channels, self.out_channels, 3, padding=1))
        
        self.xh = nn.Sequential(nn.Conv2d(self.out_channels, self.gru_hidden_channels, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.gru_hidden_channels, self.out_channels, 3, padding=1))
        
        self.hh = nn.Sequential(nn.Conv2d(self.out_channels, self.gru_hidden_channels, 3, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(self.gru_hidden_channels, self.out_channels, 3, padding=1))
    
    def initialize_weights(self, 
                           initial_func: Optional[Callable] = torch.nn.init.kaiming_normal_,
                           func_args: Optional[dict] = None,):
        if self.unet:
            self.unet.initialize_weights(initial_func, func_args)
        self.fno.initialize_weights(initial_func, func_args)

        layer_types = [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]
        for m in [self.xr, self.xz, self.xh, self.hh]:
            if any(isinstance(m, layer) for layer in layer_types):
                initial_func(m.weight.data, **func_args)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        
        if self.unet:
            x = self.unet(x)
        
        output = [x]
        
        for t in range(self.seq_len - 1):
            
            y = self.fno(x)
            z = torch.sigmoid(self.xz(y))
            r = torch.sigmoid(self.xr(y))
            h = torch.tanh(self.xh(y) + self.hh(r * x))
            x = (1 - z) * h + z * x

            output.append(x)
            
        return torch.stack(output, dim=1)

