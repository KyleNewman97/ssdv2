"""
This is an adaptation of what can be found in the ConvNeXt repo:

    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

import torch
import torch.nn as nn
from timm.layers.drop import DropPath
from torch import Tensor

from ssdv2.models.components.layer_norm import LayerNorm


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXtBlock.

    There are two equivalent implementations:

    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in
        (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU
        -> Linear; Permute back

    We use (2) as we find it slightly faster in PyTorch

    Parameters
    ----------
    dtype:
        Data type of parameters.

    device:
        The device to perform computations on.

    dim:
        Number of input channels.

    drop_path:
        Stochastic depth rate. Default: 0.0

    layer_scale_init_value:
        Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        nn.Module.__init__(self)

        # Depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dtype, device, dim, eps=1e-6)

        # Pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((dim)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.to(dtype=dtype, device=device)

    def forward(self, x: Tensor):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
