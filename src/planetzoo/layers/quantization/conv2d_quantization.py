import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional


class Conv2dQuantization(nn.Module):
    """
    N-bit quantized 2D convolutional layer that quantizes weights only.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Padding added to all four sides of the input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)
        weight_bits: Number of bits for weight quantization (default: 8)
        bias: Whether to use bias (default: True)
        signed: Whether to use signed quantization (default: True)
        freeze: Whether to freeze quantization and use as normal conv2d layer (default: False)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        weight_bits: int = 8,
        bias: bool = True,
        signed: bool = True,
        freeze: bool = False
    ):
        super(Conv2dQuantization, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.weight_bits = weight_bits
        self.signed = signed
        self.freeze = freeze
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(1))
        
        self.reset_parameters()
    
    def set_freeze(self, freeze: bool):
        """
        Set the freeze state of the quantization.
        
        Args:
            freeze: Whether to freeze quantization and use as normal conv2d layer
        """
        self.freeze = freeze
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        if self.signed:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        else:
            # For unsigned quantization, use non-negative initialization
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5, nonlinearity='relu')
            # Ensure weights are non-negative for unsigned quantization
            with torch.no_grad():
                self.weight.data = torch.abs(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
    
    def quantize(self, x: torch.Tensor, bits: int, scale: torch.Tensor) -> torch.Tensor:
        """
        Quantize tensor to n-bit representation.
        
        Args:
            x: Input tensor to quantize
            bits: Number of bits for quantization
            scale: Scale parameter for quantization
            
        Returns:
            Quantized tensor
        """
        if self.signed:
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** bits - 1
        
        # Scale and round
        x_scaled = x / scale
        x_quantized = torch.clamp(torch.round(x_scaled), qmin, qmax)
        
        # Dequantize
        return x_quantized * scale
    
    def update_scale(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        """
        Update scale parameter based on tensor statistics.
        
        Args:
            x: Input tensor
            bits: Number of bits for quantization
            
        Returns:
            Updated scale value
        """
        if self.signed:
            qmax = 2 ** (bits - 1) - 1
        else:
            qmax = 2 ** bits - 1
        
        return torch.max(torch.abs(x)) / qmax
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights or normal conv2d layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.freeze:
            # Use as normal conv2d layer without quantization (full precision)
            return F.conv2d(
                x, self.weight, self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
        else:
            # Update weight scale during training
            if self.training:
                self.weight_scale.data = self.update_scale(self.weight, self.weight_bits)
            
            # Quantize weights
            weight_quantized = self.quantize(self.weight, self.weight_bits, self.weight_scale)
            
            # Perform convolution with quantized weights
            return F.conv2d(
                x, weight_quantized, self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups
            )
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (f'{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, '
                f'stride={self.stride}, padding={self.padding}, dilation={self.dilation}, '
                f'groups={self.groups}, weight_bits={self.weight_bits}, '
                f'bias={self.bias is not None}, signed={self.signed}, freeze={self.freeze}')


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for gradient computation through quantization.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int, scale: torch.Tensor, signed: bool = True):
        if signed:
            qmin = -(2 ** (bits - 1))
            qmax = 2 ** (bits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** bits - 1
        
        x_scaled = x / scale
        x_quantized = torch.clamp(torch.round(x_scaled), qmin, qmax)
        return x_quantized * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: gradient flows unchanged
        return grad_output, None, None, None


class STEConv2dQuantization(Conv2dQuantization):
    """
    Conv2D quantization layer with Straight-Through Estimator for better gradient flow.
    """
    
    def quantize(self, x: torch.Tensor, bits: int, scale: torch.Tensor) -> torch.Tensor:
        """Quantize using straight-through estimator."""
        return StraightThroughEstimator.apply(x, bits, scale, self.signed)

