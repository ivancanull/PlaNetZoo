import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LinearQuantization(nn.Module):
    """
    N-bit quantized linear layer that quantizes weights only.
    
    Args:
        in_features: Size of input features
        out_features: Size of output features
        weight_bits: Number of bits for weight quantization (default: 8)
        bias: Whether to use bias (default: True)
        signed: Whether to use signed quantization (default: True)
        freeze: Whether to freeze quantization and use as normal linear layer (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bits: int = 8,
        bias: bool = True,
        signed: bool = True,
        freeze: bool = False
    ):
        super(LinearQuantization, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.signed = signed
        self.freeze = freeze
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(1))
        
        self.reset_parameters()
    
    def set_freeze(self, freeze: bool):
        """
        Set the freeze state of the quantization.
        
        Args:
            freeze: Whether to freeze quantization and use as normal linear layer
        """
        self.freeze = freeze
    
    def reset_parameters(self):
        """Initialize parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weight)
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
        Forward pass with quantized weights or normal linear layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.freeze:
            # Use as normal linear layer without quantization (full precision)
            return F.linear(x, self.weight, self.bias)
        else:
            # Update weight scale during training
            if self.training:
                self.weight_scale.data = self.update_scale(self.weight, self.weight_bits)
            
            # Quantize weights
            weight_quantized = self.quantize(self.weight, self.weight_bits, self.weight_scale)
            
            # Perform linear transformation with quantized weights
            return F.linear(x, weight_quantized, self.bias)
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'weight_bits={self.weight_bits}, '
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


class STELinearQuantization(LinearQuantization):
    """
    Linear quantization layer with Straight-Through Estimator for better gradient flow.
    """
    
    def quantize(self, x: torch.Tensor, bits: int, scale: torch.Tensor) -> torch.Tensor:
        """Quantize using straight-through estimator."""
        return StraightThroughEstimator.apply(x, bits, scale, self.signed)
    
    def extra_repr(self) -> str:
        """Return extra representation string for the module."""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'weight_bits={self.weight_bits}, '
                f'bias={self.bias is not None}, signed={self.signed}, freeze={self.freeze}')
