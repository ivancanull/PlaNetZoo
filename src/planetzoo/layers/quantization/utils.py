import torch.nn as nn
from .linear_quantization import LinearQuantization, STELinearQuantization
from .conv2d_quantization import Conv2dQuantization, STEConv2dQuantization


def freeze_quantization_layers(model: nn.Module, freeze: bool = True):
    """
    Freeze or unfreeze all quantization layers in a model.
    
    Args:
        model: The PyTorch model containing quantization layers
        freeze: Whether to freeze (True) or unfreeze (False) quantization layers
    """
    quantization_layers = (
        LinearQuantization, STELinearQuantization,
        Conv2dQuantization, STEConv2dQuantization
    )
    
    for module in model.modules():
        if isinstance(module, quantization_layers):
            module.set_freeze(freeze) 