import torch
import torch.nn as nn
from typing import List, Optional
from ...layers.quantization.conv2d_quantization import Conv2dQuantization, STEConv2dQuantization
from ...layers.quantization.linear_quantization import LinearQuantization, STELinearQuantization


class QuantizedBasicBlock(nn.Module):
    """
    Quantized BasicBlock for ResNet with configurable quantization parameters.
    """
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        weight_bits: int = 8,
        signed: bool = True
    ):
        super(QuantizedBasicBlock, self).__init__()
        
        self.conv1 = STEConv2dQuantization(
            in_channels, out_channels, kernel_size=3, stride=stride, 
            padding=1, bias=False, weight_bits=weight_bits, signed=signed
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = STEConv2dQuantization(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False, weight_bits=weight_bits, signed=signed
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class QuantizedResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        input_channels: int = 3,
        hidden_channels: Optional[List[int]] = None,
        blocks_per_layer: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dropout: float = 0.0,
        weight_bits: int = 8,
        signed: bool = True,
    ):
        """
        Quantized ResNet-18 with configurable channels, blocks, and quantization parameters.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
            hidden_channels: List of output channels for each layer [layer1, layer2, layer3, layer4]
                           Default: [64, 128, 256, 512]
            blocks_per_layer: List of number of blocks per layer [layer1, layer2, layer3, layer4]
                            Default: [2, 2, 2, 2] (ResNet-18 configuration)
            strides: List of strides for each layer [layer1, layer2, layer3, layer4]
                    Default: [1, 2, 2, 2]
            dropout: Dropout probability for final classifier
            weight_bits: Number of bits for weight quantization
            signed: Whether to use signed quantization
        """
        super(QuantizedResNet18, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 512]
        if blocks_per_layer is None:
            blocks_per_layer = [2, 2, 2, 2]
        if strides is None:
            strides = [1, 2, 2, 2]
        
        if len(hidden_channels) != 4:
            raise ValueError("hidden_channels must contain exactly 4 values")
        if len(blocks_per_layer) != 4:
            raise ValueError("blocks_per_layer must contain exactly 4 values")
        if len(strides) != 4:
            raise ValueError("strides must contain exactly 4 values")
        
        self.weight_bits = weight_bits
        self.signed = signed
        self.in_channels = hidden_channels[0]
        
        # Initial convolution layer
        self.conv1 = STEConv2dQuantization(
            input_channels, hidden_channels[0], kernel_size=7, stride=2,
            padding=3, bias=False, weight_bits=weight_bits, signed=signed
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(hidden_channels[0], blocks_per_layer[0], strides[0])
        self.layer2 = self._make_layer(hidden_channels[1], blocks_per_layer[1], strides[1])
        self.layer3 = self._make_layer(hidden_channels[2], blocks_per_layer[2], strides[2])
        self.layer4 = self._make_layer(hidden_channels[3], blocks_per_layer[3], strides[3])
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                STELinearQuantization(hidden_channels[3], num_classes, weight_bits=weight_bits, signed=signed)
            )
        else:
            self.classifier = STELinearQuantization(
                hidden_channels[3], num_classes, weight_bits=weight_bits, signed=signed
            )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                STEConv2dQuantization(
                    self.in_channels, out_channels, kernel_size=1, stride=stride,
                    bias=False, weight_bits=self.weight_bits, signed=self.signed
                ),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(QuantizedBasicBlock(
            self.in_channels, out_channels, stride, downsample,
            weight_bits=self.weight_bits, signed=self.signed
        ))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(QuantizedBasicBlock(
                self.in_channels, out_channels,
                weight_bits=self.weight_bits, signed=self.signed
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
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
