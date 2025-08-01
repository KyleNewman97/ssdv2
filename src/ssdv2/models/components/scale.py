import torch
from torch import Tensor, nn


class Scale(nn.Module):
    """
    Scale

    Applies a learnable scale to the input tensor.
    """

    def __init__(self, initial_value: float = 1.0):
        nn.Module.__init__(self)
        self.scale = nn.Parameter(torch.FloatTensor([initial_value]))

    def forward(self, x: Tensor):
        return x * self.scale
