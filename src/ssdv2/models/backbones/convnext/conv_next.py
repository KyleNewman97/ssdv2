"""
This is an adaptation of what can be found in the ConvNeXt repo:

    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from torch import Tensor
from torch.nn import Module

from ssdv2.models.backbones.convnext.block import Block
from ssdv2.models.backbones.convnext.layer_norm import LayerNorm


class ConvNeXt(nn.Module):
    """
    ConvNeXt
    A PyTorch impl of : `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans:
            Number of input image channels. Default: 3

        depths:
            Number of blocks at each stage. Default: [3, 3, 9, 3]

        dims:
            Feature dimension at each stage. Default: [96, 192, 384, 768]

        drop_path_rate:
            Stochastic depth rate. Default: 0.

        layer_scale_init_value:
            Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths: list[int] = [3, 3, 9, 3],  # trunk-ignore(ruff/B006)
        dims: list[int] = [96, 192, 384, 768],  # trunk-ignore(ruff/B006)
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        # Stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _init_weights(self, m: Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(self, x: Tensor) -> list[Tensor]:
        feature_maps: list[Tensor] = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            feature_maps.append(x)

        return feature_maps
