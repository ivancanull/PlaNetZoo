import torch
import torch.nn as nn
from typing import List, Optional
from ...layers.quantization.conv2d_quantization import Conv2dQuantization
from ...layers.quantization.linear_quantization import LinearQuantization


class QuantizedAlexNet(nn.Module):
    def __init__(
        self, 
        num_classes: int = 1000,
        input_channels: int = 3,
        hidden_channels: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        hstrides: Optional[List[int]] = None,
        wstrides: Optional[List[int]] = None,
        linear_sizes: Optional[List[int]] = None,
        dropout: float = 0.5,
        weight_bits: int = 8,
        quantization_backend: str = 'fbgemm'
    ):
        """
        Quantized AlexNet with configurable convolutional kernel sizes, strides, channels, and linear layer sizes.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
            hidden_channels: List of output channels for conv layers [conv1, conv2, conv3, conv4, conv5]
                           Default: [64, 192, 384, 256, 256]
            kernel_sizes: List of kernel sizes for conv layers [conv1, conv2, conv3, conv4, conv5]
                         Default: [11, 5, 3, 3, 3]
            hstrides: List of horizontal strides for conv layers [conv1, conv2, conv3, conv4, conv5]
                     Default: [4, 1, 1, 1, 1]
            wstrides: List of vertical strides for conv layers [conv1, conv2, conv3, conv4, conv5]
                     Default: [4, 1, 1, 1, 1]
            linear_sizes: List of hidden layer sizes for classifier [fc1, fc2]
                         Default: [4096, 4096]
            dropout: Dropout probability for classifier layers
            weight_bits: Number of bits for weight quantization
            quantization_backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for mobile)
        """
        super(QuantizedAlexNet, self).__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 192, 384, 256, 256]
        if kernel_sizes is None:
            kernel_sizes = [11, 5, 3, 3, 3]
        if hstrides is None:
            hstrides = [4, 1, 1, 1, 1]
        if wstrides is None:
            wstrides = [4, 1, 1, 1, 1]
        if linear_sizes is None:
            linear_sizes = [4096, 4096]
        
        if len(hidden_channels) != 5:
            raise ValueError("hidden_channels must contain exactly 5 values")
        if len(kernel_sizes) != 5:
            raise ValueError("kernel_sizes must contain exactly 5 values")
        if len(hstrides) != 5:
            raise ValueError("hstrides must contain exactly 5 values")
        if len(wstrides) != 5:
            raise ValueError("wstrides must contain exactly 5 values")
        if len(linear_sizes) != 2:
            raise ValueError("linear_sizes must contain exactly 2 values")
        
        self.weight_bits = weight_bits
        self.quantization_backend = quantization_backend
        
        # Calculate input channels for each layer
        in_channels = [input_channels] + hidden_channels[:-1]
        
        # Feature extraction layers with quantized convolutions
        self.features = nn.Sequential(
            # Conv1
            Conv2dQuantization(in_channels[0], hidden_channels[0], kernel_size=kernel_sizes[0], 
                             stride=(hstrides[0], wstrides[0]), padding=2, weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            Conv2dQuantization(in_channels[1], hidden_channels[1], kernel_size=kernel_sizes[1], 
                             stride=(hstrides[1], wstrides[1]), padding=2, weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            Conv2dQuantization(in_channels[2], hidden_channels[2], kernel_size=kernel_sizes[2], 
                             stride=(hstrides[2], wstrides[2]), padding=1, weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            
            # Conv4
            Conv2dQuantization(in_channels[3], hidden_channels[3], kernel_size=kernel_sizes[3], 
                             stride=(hstrides[3], wstrides[3]), padding=1, weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            
            # Conv5
            Conv2dQuantization(in_channels[4], hidden_channels[4], kernel_size=kernel_sizes[4], 
                             stride=(hstrides[4], wstrides[4]), padding=1, weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier layers with quantized linear layers
        final_channels = hidden_channels[-1]
        input_features = final_channels * 6 * 6
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            LinearQuantization(input_features, linear_sizes[0], weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            LinearQuantization(linear_sizes[0], linear_sizes[1], weight_bits=weight_bits),
            nn.ReLU(inplace=True),
            LinearQuantization(linear_sizes[1], num_classes, weight_bits=weight_bits),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (Conv2dQuantization, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (LinearQuantization, nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

