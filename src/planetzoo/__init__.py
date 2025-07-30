# PlaNetZoo package initialization

# Optionally, import key submodules for easier access
# from . import model, layers, processor

from . import model, layers, processor, data

# Model submodules
from .model import cnn, quantization, neuralop, simi2v, inception, utils, unet_parts, conv_lstm, unet_lstm, unet, mlp
# Layers submodules
from .layers import quantization as layers_quantization, fno_conv2d
# Processor submodules
from .processor import normalize_processor, base_processor
# Data submodules
from .data import cifar10

__all__ = [
    'model', 'layers', 'processor', 'data',
    'cnn', 'quantization', 'neuralop', 'simi2v', 'inception', 'utils', 'unet_parts', 'conv_lstm', 'unet_lstm', 'unet', 'mlp',
    'layers_quantization', 'fno_conv2d',
    'normalize_processor', 'base_processor',
    'cifar10',
]
