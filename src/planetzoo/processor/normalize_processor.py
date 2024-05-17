# TODO: implement a normalizor for multiple dimension

import torch
from typing import Sequence
from .base_processor import BaseProcessor

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

    def process(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() < max(self.dims):
            raise ValueError(f"Data dimension is less than the maximum dimension in dims: {max(self.dims)}")
        
        if self.method == 'min_max':
            self.min = torch.amin(data, dim=self.dims, keepdim=True)
            self.max = torch.amax(data, dim=self.dims, keepdim=True)
            # detach and move to cpu to save GPU memory
            self.min, self.max = self.min.detach().cpu(), self.max.detach().cpu()
            return (data - self.min) / (self.max - self.min)
        elif self.method == 'z_score':
            self.mean = torch.mean(data, dim=self.dims, keepdim=True)
            self.std = torch.std(data, dim=self.dims, keepdim=True)
            # detach and move to cpu to save GPU memory
            self.mean, self.std = self.mean.detach().cpu(), self.std.detach().cpu()
            return (data - self.mean) / self.std
        
    def restore(self, data: torch.Tensor) -> torch.Tensor:
        device = data.device
        if self.method == 'min_max':
            return data * (self.max.to(device) - self.min.to(device)) + self.min.to(device) 
        elif self.method == 'z_score':
            return data * self.std.to(device)  + self.mean.to(device) 
            