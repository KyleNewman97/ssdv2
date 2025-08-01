import math

import torch
from timm.layers.weight_init import trunc_normal_
from torch import Tensor, nn

from ssdv2.models.components import Scale


class FCOSHead(nn.Module):
    """
    FCOSHead

    A re-implementation of the FCOS head described in the paper:

    https://arxiv.org/pdf/1904.01355

    The original version can be found at:

    https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/fcos.py
    """

    def __init__(self, num_cls: int, in_channels: int, num_feature_maps: int):
        """
        Parameters
        ----------
        num_cls:
            The number of object classes to be able to predict.

        in_channels:
            The number of input channels. This must be a multiple of 32.

        num_feature_maps:
            The number of feature maps that will be passed in.
        """
        nn.Module.__init__(self)

        self.cls_path = nn.Sequential(
            *(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(),
                ]
                * 4
            ),
        )
        self.box_path = nn.Sequential(
            *(
                [
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(32, in_channels),
                    nn.ReLU(),
                ]
                * 4
            )
        )

        self.cls_logits = nn.Conv2d(in_channels, num_cls, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.apply(self._init_weights)

        # Initialise the bias for the focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)  # type: ignore

        # Initialise scales - these get applied to the regressed boxes for each feature
        # map
        self.scales = nn.ModuleList(
            [Scale(initial_value=1) for _ in range(num_feature_maps)]
        )

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(
        self, feature_maps: list[Tensor] | Tensor
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        logits: list[Tensor] = []
        box_reg: list[Tensor] = []
        centerness: list[Tensor] = []

        if isinstance(feature_maps, Tensor):
            feature_maps = [feature_maps]

        for idx, fm in enumerate(feature_maps):
            cls_out: Tensor = self.cls_path(fm)
            box_out: Tensor = self.box_path(fm)

            logits.append(self.cls_logits(cls_out))
            centerness.append(self.centerness(cls_out))

            box_pred: Tensor = self.box_pred(box_out)
            box_pred = self.scales[idx](box_pred)
            box_reg.append(torch.exp(box_pred))

        return logits, box_reg, centerness
