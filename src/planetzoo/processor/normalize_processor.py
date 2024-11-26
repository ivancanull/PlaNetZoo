# TODO: implement a normalizor for multiple dimension

import torch
from typing import Sequence
from .base_processor import BaseProcessor

__all__ = ["NormalizeProcessor"]

class NormalizeProcessor(BaseProcessor):
    def __init__(self, 
                 dims: Sequence[int],
                 method: str = 'z_score',):
        """
        A processor to normalize data along specified dimensions.
        
        Args:
            dims (Sequence[int]): Dimensions to normalize.
            method (str, optional): Normalization method. Defaults to 'z_score'. Support methods: ['min_max', 'z_score'].
        
        """
        
        self.dims = dims
        self.method = method
        if self.method not in ['min_max', 'z_score']:
            raise ValueError(f"Method {self.method} is not supported!")

    def process(self, data: torch.Tensor, initialize=False) -> torch.Tensor:
        if data.dim() < max(self.dims):
            raise ValueError(f"Data dimension is less than the maximum dimension in dims: {max(self.dims)}")
        device = data.device
        
        if self.method == 'min_max':
            if initialize:
                self.min = torch.amin(data, dim=self.dims, keepdim=True).detach().cpu() # detach and move to cpu to save GPU memory
                self.max = torch.amax(data, dim=self.dims, keepdim=True).detach().cpu()
            return (data - self.min.to(device)) / (self.max.to(device) - self.min.to(device))
        elif self.method == 'z_score':
            if initialize:
                self.mean = torch.mean(data, dim=self.dims, keepdim=True).detach().cpu()
                self.std = torch.std(data, dim=self.dims, keepdim=True).detach().cpu()
            return (data - self.mean.to(device)) / self.std.to(device)

    def restore(self, data: torch.Tensor) -> torch.Tensor:
        device = data.device
        if self.method == 'min_max':
            return data * (self.max.to(device) - self.min.to(device)) + self.min.to(device) 
        elif self.method == 'z_score':
            return data * self.std.to(device)  + self.mean.to(device) 
                