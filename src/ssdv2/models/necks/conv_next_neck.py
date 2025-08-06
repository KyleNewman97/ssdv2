import torch
from timm.layers.weight_init import trunc_normal_
from torch import Tensor, nn

from ssdv2.models.components import ConvNeXtBlock
from ssdv2.structs import FeatureMap


class ConvNeXtNeck(nn.Module):
    """
    ConvNeXtNeck

    A object detector neck implemented with ConvNeXt blocks.
    """

    def __init__(
        self,
        depths: list[int] = [0, 3, 3, 3],  # trunk-ignore(ruff/B006)
        fm_channels: list[int] = [384, 192, 96, 48],  # trunk-ignore(ruff/B006)
    ):
        """
        Parameters
        ----------
        depths:
            The number of convolutional layers in each ConvNeXtBlock.

        dims:
            Feature map dimensions at each stage.
        """
        nn.Module.__init__(self)

        # Ensure that the depths and dims have the same length
        if len(depths) != len(fm_channels):
            msg = f"len(depths) != len(dims) -> {len(depths)} != {len(fm_channels)}"
            raise ValueError(msg)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Create the network stages
        self.num_stages = len(depths)
        self.stages = nn.ModuleList()
        in_channels = 0
        out_channels = 0
        self.out_fm_channels: list[int] = []
        for i in range(self.num_stages):
            # Calculate the number of input and output channels
            in_channels = out_channels + fm_channels[-(i + 1)]
            out_channels = out_channels + fm_channels[-(i + 1)] // 2
            self.out_fm_channels.append(out_channels)

            # Construct the stage
            stage = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                *[ConvNeXtBlock(dim=out_channels) for _ in range(depths[i])],
            )
            self.stages.append(stage)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(self, feature_maps: list[FeatureMap]) -> list[FeatureMap]:
        reversed_fms = feature_maps[::-1]

        upsampled_fm_data: Tensor
        out_fms: list[FeatureMap] = []
        for idx, fm in enumerate(reversed_fms):
            if idx == 0:
                fm_data = fm.data
            else:
                # trunk-ignore(ruff/F821)
                fm_data = torch.cat((upsampled_fm_data, fm.data), dim=1)  # type: ignore

            out_fm_data = self.stages[idx].forward(fm_data)
            out_fms.append(
                FeatureMap(
                    data=out_fm_data,
                    stride=fm.stride,
                    index=idx,
                    all_strides=fm.all_strides[::-1],
                )
            )

            upsampled_fm_data = self.upsample(out_fm_data)

        return out_fms
