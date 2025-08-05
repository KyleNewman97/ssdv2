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
        depths: list[int] = [3, 3, 3],  # trunk-ignore(ruff/B006)
        dims: list[int] = [576, 672, 720],  # trunk-ignore(ruff/B006)
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
        if len(depths) != len(dims):
            msg = f"len(depths) != len(dims) -> {len(depths)} != {len(dims)}"
            raise ValueError(msg)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Create the network stages
        self.num_stages = len(depths)
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])],
            )
            self.stages.append(stage)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(self, feature_maps: list[FeatureMap]) -> list[FeatureMap]:
        reversed_fms = feature_maps[::-1]

        out_fms: list[FeatureMap] = [reversed_fms[0]]
        out_fms[0].index = 0
        out_fms[0].all_strides = out_fms[0].all_strides[::-1]
        upsampled_fm_data: Tensor = self.upsample(reversed_fms[0].data)

        for idx, fm in enumerate(reversed_fms[1:]):
            fm_data = torch.cat((upsampled_fm_data, fm.data), dim=1)
            out_fm_data: Tensor = self.stages[idx].forward(fm_data)
            out_fms.append(
                FeatureMap(
                    data=out_fm_data,
                    stride=fm.stride,
                    index=idx + 1,
                    all_strides=fm.all_strides[::-1],
                )
            )

            upsampled_fm_data = self.upsample(out_fm_data)

        return out_fms
