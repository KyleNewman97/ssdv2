"""
This is an adaptation of what can be found in the ConvNeXt repo:

    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
"""

import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_
from torch import Tensor

from ssdv2.models.components import ConvNeXtBlock, LayerNorm
from ssdv2.structs import FeatureMap


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXtBackbone

    A PyTorch impl of : `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
        self,
        depths: list[int] = [3, 3, 9, 3],  # trunk-ignore(ruff/B006)
        dims: list[int] = [96, 192, 384, 768],  # trunk-ignore(ruff/B006)
        strides: list[int] = [4, 2, 2, 2],  # trunk-ignore(ruff/B006)
        in_chans: int = 3,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        """
        Parameters
        ----------
        depths:
            Number of blocks at each stage. Default: [3, 3, 9, 3]

        dims:
            Feature dimension at each stage. Default: [96, 192, 384, 768]

        strides:
            The stride to apply after each each stage. This acts as the downsampling
            factor.

        in_chans:
            Number of input image channels. Default: 3

        drop_path_rate:
            Stochastic depth rate. Default: 0.

        layer_scale_init_value:
            Init value for Layer Scale. Default: 1e-6.
        """
        nn.Module.__init__(self)

        # Check the number of depths, dims and strides are equal
        if len(depths) != len(dims) or len(dims) != len(strides):
            raise ValueError("Depths, dims and strides must all have the same length.")
        self.num_stages = len(depths)

        # Stem and (num_stages - 1) intermediate downsampling conv layers
        self.strides = strides
        self.downsample_layers = nn.ModuleList()
        stride = self.strides[0]
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stride, stride=stride),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stages - 1):
            stride = self.strides[i + 1]
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=stride, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)

        # Feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
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

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(self, x: Tensor) -> list[FeatureMap]:
        # Calculate the cumulative stride at each level
        cumulative_stride = 0
        cumulative_strides: list[int] = []
        for i in range(self.num_stages):
            if cumulative_stride == 0:
                cumulative_stride = self.strides[i]
            elif self.strides[i] != 0:
                cumulative_stride *= self.strides[i]
            cumulative_strides.append(cumulative_stride)

        feature_maps: list[FeatureMap] = []
        for i in range(self.num_stages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

            feature_maps.append(
                FeatureMap(
                    data=x,
                    stride=cumulative_strides[i],
                    index=i,
                    all_strides=cumulative_strides,
                )
            )

        return feature_maps
