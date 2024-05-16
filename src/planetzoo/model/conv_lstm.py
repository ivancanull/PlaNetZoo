# Implementation of Convolutional LSTM model.
# Shi, Xingjian, et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in neural information processing systems 28 (2015).
# https://arxiv.org/abs/1506.04214

# Reference Codes:
# https://github.com/ndrplz/ConvLSTM_pytorch

import torch
import torch.nn as nn

from typing import List, Optional, Callable, Tuple

__all__ = ["ConvLSTM"]


class ConvLSTMCell(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 kernel_size: Optional[int] = 3, 
                 padding: Optional[int] = 1, 
                 bias: bool = True):
        """
        
        Input Features:
            x: (batch_size, in_channels, height, width)
            h: (batch_size, hidden_channels, height, width)
            c: (batch_size, hidden_channels, height, width)
        
        Output Features:
            h: (batch_size, hidden_channels, height, width)
            c: (batch_size, hidden_channels, height, width)
        
        Operations:
            i = sigmoid(W_ii * x + W_hi * h + b_ii + b_hi)
            f = sigmoid(W_if * x + W_hf * h + b_if + b_hf)
            o = sigmoid(W_io * x + W_ho * h + b_io + b_ho)
            g = tanh(W_ig * x + W_hg * h + b_ig + b_hg)
            c = f * c + i * g
            h = o * tanh(c)
        
        
        """
        super().__init__()

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(in_channels=in_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
    
    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=-3)

        x = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.hidden_channels, dim=-3)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))


class ConvLSTMCell(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 kernel_size: Optional[int] = 3, 
                 padding: Optional[int] = 1, 
                 bias: bool = True):
        """
        
        Input Features:
            x: (batch_size, in_channels, height, width)
            h: (batch_size, hidden_channels, height, width)
            c: (batch_size, hidden_channels, height, width)
        
        Output Features:
            h: (batch_size, hidden_channels, height, width)
            c: (batch_size, hidden_channels, height, width)
        
        Operations:
            i = sigmoid(W_ii * x + W_hi * h + b_ii + b_hi)
            f = sigmoid(W_if * x + W_hf * h + b_if + b_hf)
            o = sigmoid(W_io * x + W_ho * h + b_io + b_ho)
            g = tanh(W_ig * x + W_hg * h + b_ig + b_hg)
            c = f * c + i * g
            h = o * tanh(c)
        
        
        """
        super().__init__()

        self.hidden_channels = hidden_channels

        self.conv = nn.Conv2d(in_channels=in_channels + hidden_channels,
                              out_channels=4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
    
    def forward(self, x, h, c):

        combined = torch.cat([x, h], dim=-3)

        x = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(x, self.hidden_channels, dim=-3)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: List[int], 
                 kernel_sizes: Optional[List[int]] = None, 
                 paddings: Optional[List[int]] = None,
                 batch_first=False, 
                 bias=True, 
                 return_all_layers=False):
        
        """
        
        Input Features:
            (batch_size, seq_len, in_channels, height, width) or (seq_len, batch_size, in_channels, height, width)
        
        Output Features:
            (batch_size, seq_len, hidden_channels[-1], height, width) or (seq_len, batch_size, hidden_channels[-1], height, width)

        Operations:
            h, c = init_hidden(batch_size, image_size)
            for t in range(seq_len):
                for i in range(num_layers):
                    h, c = layers[i](x=h, cur_state=[h, c])
        """
        super().__init__()


        if kernel_sizes is None:
            kernel_sizes = [3] * len(hidden_channels)
        if paddings is None:
            paddings = [1] * len(hidden_channels)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = len(self.hidden_channels)
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

     
        layers = []
        in_dim = in_channels
        for i in range(len(hidden_channels)):
            hidden_dim = hidden_channels[i]
            kernel_size = kernel_sizes[i]
            padding = paddings[i]
            layers.append(ConvLSTMCell(in_channels=in_dim,
                                       hidden_channels=hidden_dim,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       bias=bias,
                                       ))
            in_dim = hidden_dim
        self.conv_lstm = nn.Sequential(*layers)

    def forward(self, 
                x: torch.Tensor, 
                hidden_state: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            x = x.permute(1, 0, 2, 3, 4)

        batch_size, seq_len, in_channels, rows, cols = x.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=batch_size,
                                             image_size=(rows, cols))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.conv_lstm[layer_idx](x=cur_layer_input[:, t, :, :, :],
                                                 h=h,
                                                 c=c)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.conv_lstm[i].init_hidden(batch_size, image_size))
        return init_states


if __name__ == "__main__":
    # test example
    device = torch.device("cuda:0")
    model = ConvLSTM(in_channels=1, hidden_channels=[64, 64, 128, 6], batch_first=True, return_all_layers=False).to(device)
    x = torch.randn((3, 10, 1, 56, 42)).to(device)
    layer_output_list, last_state_list = model(x)
    print(layer_output_list[-1].shape) # x
    print(last_state_list[-1][0].shape) # h