from typing import Callable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

__all__ = ["MLP"]

class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) with hidden layers"""
    def __init__(self,
                     in_channels: int,
                     hidden_channels: List[int],
                     norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                     activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
                     inplace: Optional[bool] = None,
                     bias: bool = True,):
        """
        
        Args:
            in_channels (int): Number of input channels.
            hidden_channels (List[int]): List of hidden layer dimensions.
            norm_layer (Optional[Callable[..., torch.nn.Module]], optional): Normalization layer to be applied after each hidden layer. Defaults to None.
            activation_layer (Optional[Callable[..., torch.nn.Module]], optional): Activation function to be applied after each hidden layer. Defaults to torch.nn.ReLU.
            inplace (Optional[bool], optional): Whether to perform the activation function in-place. Defaults to None.
            bias (bool, optional): Whether to include bias in the linear layers. Defaults to True.
        
        Input Features: 
            (batch_size, in_channels)
        Output Features:
            (batch_size, hidden_channels[-1])
        
        """
        super().__init__()

        layers = []
        in_dim = in_channels
        params = {} if inplace is None else {"inplace": inplace}
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        self.layers = nn.Sequential(*layers)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        y = self.layers(x)
        return y