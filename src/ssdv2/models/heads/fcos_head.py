import math

import torch
from timm.layers.weight_init import trunc_normal_
from torch import Tensor, nn
from torchvision.ops import box_convert

from ssdv2.models.components import Scale
from ssdv2.structs import FeatureMap, FrameLabels


class FCOSHead(nn.Module):
    """
    FCOSHead

    A re-implementation of the FCOS head described in the paper:

    https://arxiv.org/pdf/1904.01355

    The original version can be found at:

    https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/fcos.py

    It should be noted, the original paper re-used the same head for each feature map.
    However, this requires feature map containing the same number of channels. To avoid
    this limitation this head has been adapted to only work with a single feature map.
    """

    def __init__(self, num_cls: int, in_channels: int):
        """
        Parameters
        ----------
        num_cls:
            The number of object classes to be able to predict.

        in_channels:
            The number of input channels. This must be a multiple of 32.
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

        # Initialise the scale - this gets applied to the regressed bounding boxes
        self.scale = Scale(initial_value=1)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(self, feature_map: FeatureMap) -> tuple[Tensor, Tensor, Tensor]:
        # Run the feature map through the class and box paths
        cls_out: Tensor = self.cls_path(feature_map.data)
        box_out: Tensor = self.box_path(feature_map.data)

        # Predict the logits and centerness
        logits = self.cls_logits(cls_out)
        centerness = self.centerness(cls_out)

        # Predict the boxes
        box_pred: Tensor = self.box_pred(box_out)
        box_pred = self.scale(box_pred)
        box_reg = torch.exp(box_pred)

        return logits, box_reg, centerness

    @staticmethod
    def calculate_feature_map_locations(
        feature_map: FeatureMap,
    ) -> Tensor:
        """
        Find the location of each feature map pixel within the original image. The
        conversion from feature map pixel location to image location is done with:

        `(s/2 + x*s, s/2 + y*s)`

        where:
            `s` - cumulative stride used by the model in producing this feature map.
            `x` - x pixel location in the feature map.
            `y` - y pixel location in the feature map.

        Parameters
        ----------
        feature_map:
            The feature map to find image locations for.

        Returns
        -------
        indices:
            The (x, y) indices of every location of the feature map in the original
            image. This should have a shape of `(fm_height*fm_width, 2)`.
        """
        stride = feature_map.stride
        x_locs = torch.arange(0, feature_map.width * stride, stride) + stride // 2
        y_locs = torch.arange(0, feature_map.height * stride, stride) + stride // 2
        x_indices, y_indices = torch.meshgrid(x_locs, y_locs, indexing="xy")

        return torch.stack((x_indices.reshape(-1), y_indices.reshape(-1)), dim=1)

    @staticmethod
    def filter_objects_by_feature_map(
        feature_map: FeatureMap, objects: FrameLabels
    ) -> FrameLabels:
        """
        Removes objects that should not be associated with the input feature map. This
        implements the "multi-level prediction" outlined in the FCOS paper.
        Specifically, this only keeps objects within the size bound supported by the
        feature map.

        Parameters
        ----------
        feature_map:
            The feature map we want to find objects for.

        objects:
            All objects associated with the frame.

        Returns
        -------
        feature_map_objects:
            Objects that should be associated with this feature map.
        """
        # Find the minimum object width and height this feature map can support
        min_width = feature_map.fcos_min_object_width
        min_height = feature_map.fcos_min_object_height

        # Find the maximum object width and height this feature map can support
        max_width = feature_map.fcos_max_object_width
        max_height = feature_map.fcos_max_object_height

        # Find the image location of each pixel in the feature map
        image_locations_xy = FCOSHead.calculate_feature_map_locations(feature_map)

        # Convert boxes to xyxy and in image pixel coords
        object_boxes_xyxy = box_convert(objects.boxes, "cxcywh", "xyxy")
        object_boxes_xyxy[:, ::2] *= feature_map.image_width
        object_boxes_xyxy[:, 1::2] *= feature_map.image_height
        class_ids = objects.class_ids

        num_locations = image_locations_xy.shape[0]
        for object_box_xyxy, class_id in zip(object_boxes_xyxy, class_ids, strict=True):
            # Calculate all regression targets for this box
            dupe_box_xyxy = object_box_xyxy.expand((num_locations, -1))
            l = image_locations_xy[:, 0] - dupe_box_xyxy[:, 0]
            t = image_locations_xy[:, 1] - dupe_box_xyxy[:, 1]
            r = dupe_box_xyxy[:, 2] - image_locations_xy[:, 0]
            b = dupe_box_xyxy[:, 3] - image_locations_xy[:, 1]
            regression_targets = torch.stack([l, t, r, b], dim=1)

            # Determine if any meet the size requirements
            mask = min_width <= regression_targets[:, ::2].min(dim=1).values
            mask &= regression_targets[:, ::2].max(dim=1).values < max_width
            mask &= min_height <= regression_targets[:, 1::2].min(dim=1).values
            mask &= regression_targets[:, 1::2].max(dim=1).values < max_height

            # Calculate the class targets for this box
            cls_targets = torch.zeros(
                (num_locations,), dtype=torch.int, device=mask.device
            )
            cls_targets[mask] = class_id

        # Find boxes that meet the size condition
        size_mask = min_width <= objects.boxes[:, 2]
        size_mask = size_mask.bitwise_and(objects.boxes[:, 2] < max_width)
        size_mask = size_mask.bitwise_and(min_height < objects.boxes[:, 3])
        size_mask = size_mask.bitwise_and(objects.boxes[:, 3] < max_height)
        boxes = objects.boxes[size_mask, :]
        class_ids = objects.raw_class_ids[size_mask]

        return FrameLabels(
            boxes=boxes,
            raw_class_ids=class_ids,
            raw_class_names=objects.raw_class_names,
        )

    @staticmethod
    def loss(logits: Tensor, box_regs: Tensor, centerness: Tensor, gt_boxes: Tensor):
        pass
