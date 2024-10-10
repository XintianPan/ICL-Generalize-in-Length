from typing import Any, Dict, Optional
from dataset_base import DataMethod
from samplers import sample_transformation
import torch
import math

class LinearReg(DataMethod):
    """
    A data method that generates linear regression data given data size, 
    sequence length, noise level, and the condition number of the covariance 
    matrix of the data.

    Attributes:
        dict (Dict): The parameters for the data generation.
    """

    def __init__(self, dict: Dict = None):
        """
        Initialize the LinearReg data method.

        Args:
            dict (Dict): The parameters for the data generation.
        """
        super().__init__(dict)
        self.L = dict['L']
        self.dx = dict['dx']
        self.dy = dict['dy']
        self.d = self.dx + self.dy
        self.noise_std = dict['noise_std']
        self.number_of_samples = dict['number_of_samples']
        
        # initialize G matrix indexing each task's position
        self.G = torch.zeros(self.dx, self.dy)
        self.r = self.dx // self.dy
        for i in range(self.dy):
            self.G[int(i * self.r):int((i + 1) * self.r), i] = 1

    def __generatedata__(self, **kwargs) -> Any:
        """
        Generate linear regression data.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The generated data.
        """

        # generate x and x_q
        x = torch.randn(self.number_of_samples, self.L, self.dx)    # (n, L, dx)
        # x_q = torch.randn(self.number_of_samples, 1, self.dx)    # (n, 1, dx)

        # generate beta
        beta = torch.randn(self.number_of_samples, self.dx, self.dy)    # (n, dx, dy)
        beta = torch.einsum('nxy,xy->nxy', beta, self.G)    # (n, dx, dy)
        
        # generate y
        y = torch.einsum('nlx,nxy->nly', x, beta)   # (n, L, dy)
        y += (math.sqrt(self.dx) * self.noise_std 
              * torch.randn(self.number_of_samples, self.L, self.dy))
        # y_q = torch.einsum('nlx,nxy->nly', x_q, beta)  # (n, 1, dy)

        # generate z by concatenating x and y
        z = torch.cat([x, y], dim = 2)
        # z_q = torch.cat([x_q, torch.zeros_like(y_q)], dim = 2)
        return z.squeeze(0)

    def __transform__(self, x: Any, zero_index: Optional[int] = None, **kwargs) -> Any:
        """
        Transform the data for training, validation, and testing.

        Args:
            x (Any): The data.

        Returns:
            Any: The transformed data.
        """
        y = x[..., :, -1].clone()

        if zero_index is not None:
            x[..., zero_index, -1] = 0

        return x, y
