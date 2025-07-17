# PlaNetZoo 🦁

A comprehensive zoo for neural network implementations, focusing on computer vision and spatiotemporal modeling architectures.

## 🌟 Features

- **Vision Models**: U-Net, AlexNet, and variants with LSTM integration
- **Spatiotemporal Models**: ConvLSTM, SimI2V (Simple Image-to-Video), FNO-based models
- **Neural Operators**: Fourier Neural Operators (FNO) for PDE solving
- **Quantization Layers**: Linear quantization with Straight-Through Estimator
- **Data Processing**: Normalization processors for multi-dimensional data
- **Modular Design**: Easy-to-extend components for rapid prototyping

## 📦 Installation

### From Source

```bash
git clone https://github.com/ivancanull/PlaNetZoo.git
cd PlaNetZoo
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch
- NumPy
- PrettyTable (for model parameter counting)

## 🚀 Quick Start

### Basic U-Net for Image Segmentation

```python
import torch
from planetzoo.model import UNet

# Create U-Net model
model = UNet(
    in_channels=3,
    hidden_channels=[64, 128, 256, 512],
    out_channels=1
)

# Initialize weights
model.initialize_weights()

# Forward pass
x = torch.randn(4, 3, 256, 256)  # Batch of RGB images
output = model(x)  # Shape: (4, 1, 256, 256)
```

### ConvLSTM for Video Prediction

```python
from planetzoo.model import ConvLSTM

# Create ConvLSTM model
model = ConvLSTM(
    in_channels=3,
    hidden_channels=[64, 64, 128],
    batch_first=True,
    return_all_layers=False
)

# Forward pass
x = torch.randn(4, 10, 3, 64, 64)  # (batch, time, channels, height, width)
layer_outputs, last_states = model(x)
predictions = layer_outputs[-1]  # Shape: (4, 10, 128, 64, 64)
```

### Image-to-Video Generation with SimI2V

```python
from planetzoo.model import SimI2V

# Create SimI2V model
model = SimI2V(
    in_channels=1,
    hidden_channels=[32, 64, 128, 256, 128],
    out_channels=6,
    seq_len=25,
    groups=3
)

# Generate video from single image
x = torch.randn(4, 1, 56, 42)  # Single frame input
video = model(x)  # Shape: (4, 25, 6, 56, 42) - 25 frames output
```

### Data Normalization

```python
from planetzoo.processor import NormalizeProcessor

# Create processor for z-score normalization
processor = NormalizeProcessor(
    dims=[0, 2, 3],  # Normalize across batch and spatial dimensions
    method='z_score'
)

# Process data
data = torch.randn(32, 3, 64, 64)
normalized = processor.process(data, initialize=True)

# Restore original scale
restored = processor.restore(normalized)
```

### Quantized Neural Networks

```python
from planetzoo.layers.quantization import LinearQuantization, STELinearQuantization

# Standard quantized linear layer
quant_layer = LinearQuantization(
    in_features=128,
    out_features=64,
    weight_bits=4,
    signed=True
)

# With Straight-Through Estimator for better gradients
ste_layer = STELinearQuantization(
    in_features=128,
    out_features=64,
    weight_bits=8
)
```

## 🏗️ Architecture Overview

### Model Categories

#### Vision Models (`planetzoo.model`)
- **UNet**: Encoder-decoder architecture for image segmentation
- **UNetLSTM**: U-Net with LSTM integration for temporal modeling
- **AlexNet**: Configurable AlexNet implementation

#### Spatiotemporal Models
- **ConvLSTM**: Convolutional LSTM for video sequence modeling
- **SimI2V**: Simple Image-to-Video prediction models
- **FNO2D**: Fourier Neural Operator variants (2D, RNN, GRU)

#### Building Blocks (`planetzoo.layers`)
- **FNOConv2d**: Fourier convolution layer
- **Quantization**: Linear quantization layers

### Processing Pipeline (`planetzoo.processor`)
- **NormalizeProcessor**: Multi-dimensional data normalization
- **BaseProcessor**: Abstract base class for processors

## 📊 Model Utilities

### Parameter Counting

```python
from planetzoo.model.utils import count_parameters

model = UNet(in_channels=3, hidden_channels=[64, 128, 256, 512], out_channels=1)
total_params = count_parameters(model)
# Outputs a formatted table with layer-wise parameter counts
```

### Weight Initialization

```python
# Kaiming initialization
model.initialize_weights(
    initial_func=torch.nn.init.kaiming_uniform_,
    func_args={"mode": "fan_in", "nonlinearity": "relu"}
)

# Xavier initialization
model.initialize_weights(
    initial_func=torch.nn.init.xavier_normal_,
    func_args={}
)
```

## 🧪 Testing

Run the quantization layer tests:

```bash
python test_runner.py
```

Or use pytest for more detailed testing:

```bash
pytest tests/ -v
```

## 📖 API Reference

### Core Models

#### UNet
```python
UNet(
    in_channels: int,
    hidden_channels: List[int],  # Must have 4 elements
    out_channels: int,
    norm_layer: Optional[Callable] = None,
    activation_layer: Optional[Callable] = torch.nn.ReLU,
    bilinear: bool = False
)
```

#### ConvLSTM
```python
ConvLSTM(
    in_channels: int,
    hidden_channels: List[int],
    kernel_sizes: Optional[List[int]] = None,
    batch_first: bool = False,
    return_all_layers: bool = False
)
```

#### SimI2V
```python
SimI2V(
    in_channels: int,
    hidden_channels: List[int],  # Must have 5 elements
    out_channels: int,
    seq_len: int,
    groups: int,
    norm_layer: Optional[Callable] = None,
    activation_layer: Optional[Callable] = torch.nn.ReLU
)
```

### Quantization Layers

#### LinearQuantization
```python
LinearQuantization(
    in_features: int,
    out_features: int,
    weight_bits: int = 8,
    bias: bool = True,
    signed: bool = True
)
```

### Processors

#### NormalizeProcessor
```python
NormalizeProcessor(
    dims: Sequence[int],
    method: str = 'z_score'  # 'z_score' or 'min_max'
)
```

## 🎯 Use Cases

- **Medical Image Segmentation**: Use U-Net for organ/tissue segmentation
- **Video Prediction**: ConvLSTM and SimI2V for future frame prediction
- **Weather Forecasting**: FNO models for solving meteorological PDEs
- **Satellite Imagery**: Temporal analysis with UNet-LSTM
- **Edge Deployment**: Quantized models for resource-constrained environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SimVP**: Inspiration for SimI2V architecture
- **U-Net**: Original paper by Ronneberger et al.
- **ConvLSTM**: Shi et al. implementation
- **FNO**: Li et al. Fourier Neural Operator
- **UNet-LSTM**: Papadomanolaki et al. implementation

## 📞 Contact

- **Author**: Ivan Canull
- **Email**: ivancanull@gmail.com
- **GitHub**: [ivancanull](https://github.com/ivancanull/PlaNetZoo)

---

**PlaNetZoo**: Where neural networks roam free! 🦁🧠
