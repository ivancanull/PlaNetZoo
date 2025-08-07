import torch
import torch.nn as nn
from typing import List, Optional
from ...layers.quantization.conv2d_quantization import Conv2dQuantization, STEConv2dQuantization
from ...layers.quantization.linear_quantization import LinearQuantization, STELinearQuantization


class QuantizedDepthwiseSeparableConv(nn.Module):
    """
    Quantized Depthwise Separable Convolution block for MobileNet.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        weight_bits: int = 8,
        signed: bool = True
    ):
        super(QuantizedDepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = STEConv2dQuantization(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
            weight_bits=weight_bits, signed=signed
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Pointwise convolution
        self.pointwise = STEConv2dQuantization(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False, weight_bits=weight_bits, signed=signed
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class QuantizedMobileNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        input_channels: int = 3,
        width_multiplier: float = 1.0,
        hidden_channels: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dropout: float = 0.2,
        weight_bits: int = 8,
        signed: bool = True,
    ):
        """
        Quantized MobileNet with configurable channels, strides, and quantization parameters.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
            width_multiplier: Width multiplier to scale number of channels
            hidden_channels: List of output channels for each depthwise separable block
                           Default: [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
            strides: List of strides for each layer
                    Default: [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
            dropout: Dropout probability for final classifier
            weight_bits: Number of bits for weight quantization
            signed: Whether to use signed quantization
        """
        super(QuantizedMobileNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        if strides is None:
            strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
        
        if len(hidden_channels) != 14:
            raise ValueError("hidden_channels must contain exactly 14 values")
        if len(strides) != 13:
            raise ValueError("strides must contain exactly 13 values")
        
        self.weight_bits = weight_bits
        self.signed = signed
        
        # Apply width multiplier and ensure channels are divisible by 8
        def _make_divisible(ch, divisor=8):
            return int((ch * width_multiplier + divisor / 2) // divisor) * divisor
        
        # Scale hidden channels by width multiplier
        scaled_channels = [_make_divisible(ch) for ch in hidden_channels]
        
        # Initial convolution layer
        self.conv1 = STEConv2dQuantization(
            input_channels, scaled_channels[0], kernel_size=3, stride=2,
            padding=1, bias=False, weight_bits=weight_bits, signed=signed
        )
        self.bn1 = nn.BatchNorm2d(scaled_channels[0])
        self.relu1 = nn.ReLU(inplace=True)
        
        # Depthwise separable convolution layers
        self.features = nn.ModuleList()
        in_ch = scaled_channels[0]
        
        for i, (out_ch, stride) in enumerate(zip(scaled_channels[1:], strides)):
            self.features.append(
                QuantizedDepthwiseSeparableConv(
                    in_ch, out_ch, stride=stride,
                    weight_bits=weight_bits, signed=signed
                )
            )
            in_ch = out_ch
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                STELinearQuantization(scaled_channels[-1], num_classes, weight_bits=weight_bits, signed=signed)
            )
        else:
            self.classifier = STELinearQuantization(
                scaled_channels[-1], num_classes, weight_bits=weight_bits, signed=signed
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        for layer in self.features:
            x = layer(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (Conv2dQuantization, STEConv2dQuantization, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LinearQuantization, STELinearQuantization, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
