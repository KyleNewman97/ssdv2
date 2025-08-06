import math

import torch
from timm.layers.weight_init import trunc_normal_
from torch import Tensor, nn
from torch.nn.functional import one_hot
from torchvision.ops import (
    box_area,
    box_convert,
    complete_box_iou_loss,
    sigmoid_focal_loss,
)

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

    def __init__(self, num_cls: int, in_channels: list[int]):
        """
        Parameters
        ----------
        num_cls:
            The number of object classes to be able to predict.

        in_channels:
            The number of input channels per feature map. Each one if these should be a
            multiple of 32.
        """
        nn.Module.__init__(self)
        self.num_cls = num_cls

        self.cls_path_per_fm = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        [
                            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                            nn.GroupNorm(32, channels),
                            nn.ReLU(),
                        ]
                        * 4
                    ),
                )
                for channels in in_channels
            ]
        )
        self.box_path_per_fm = nn.ModuleList(
            [
                nn.Sequential(
                    *(
                        [
                            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                            nn.GroupNorm(32, channels),
                            nn.ReLU(),
                        ]
                        * 4
                    )
                )
                for channels in in_channels
            ]
        )

        self.cls_logits_per_fm = nn.ModuleList(
            [
                nn.Conv2d(channels, num_cls, kernel_size=3, padding=1)
                for channels in in_channels
            ]
        )
        self.box_pred_per_fm = nn.ModuleList(
            [
                nn.Conv2d(channels, 4, kernel_size=3, padding=1)
                for channels in in_channels
            ]
        )
        self.centerness_per_fm = nn.ModuleList(
            [
                nn.Conv2d(channels, 1, kernel_size=3, padding=1)
                for channels in in_channels
            ]
        )

        self.apply(self._init_weights)

        # Initialise the bias for the focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for cls_logits in self.cls_logits_per_fm:
            nn.init.constant_(cls_logits.bias, bias_value)  # type: ignore

        # Initialise the scale - this gets applied to the regressed bounding boxes
        self.scale = Scale(initial_value=1)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward(self, feature_maps: list[FeatureMap]) -> tuple[Tensor, Tensor, Tensor]:
        all_logits = []
        all_centerness = []
        all_boxes = []
        for fm in feature_maps:
            # Run the feature map through the class and box paths
            cls_path_out: Tensor = self.cls_path_per_fm[fm.index](fm.data)
            box_path_out: Tensor = self.box_path_per_fm[fm.index](fm.data)

            # Predict the logits and centerness
            logits: Tensor = self.cls_logits_per_fm[fm.index](cls_path_out)
            centerness: Tensor = self.centerness_per_fm[fm.index](box_path_out)

            # Predict the boxes
            box_predictions: Tensor = self.box_pred_per_fm[fm.index](box_path_out)
            box_predictions = self.scale(box_predictions)
            boxes_delta = torch.exp(box_predictions)

            # Convert boxes from delta domain to image domain
            locations = FCOSHead.calculate_feature_map_locations(fm)
            boxes_image = FCOSHead.delta_to_image_domain(boxes_delta, locations)

            # Convert shape to (batch, h*w, d)
            batch_size = logits.shape[0]
            logits = logits.permute(0, 2, 3, 1)
            logits = logits.reshape(batch_size, -1, self.num_cls)
            centerness = centerness.permute(0, 2, 3, 1)
            centerness = centerness.reshape(batch_size, -1, 1)
            boxes_image = boxes_image.permute(0, 2, 3, 1)
            boxes_image = boxes_image.reshape(batch_size, -1, 4)

            all_logits.append(logits)
            all_centerness.append(centerness)
            all_boxes.append(boxes_image)

        all_logits = torch.cat(all_logits, dim=1)
        all_centerness = torch.cat(all_centerness, dim=1)
        all_boxes = torch.cat(all_boxes, dim=1)

        return all_logits, all_centerness, all_boxes

    @staticmethod
    def delta_to_image_domain(boxes_delta: Tensor, locations: Tensor) -> Tensor:
        """
        Converts boxes from the delta domain `(l^*, t^*, r^*, b^*)` to the image
        domain `(l, t, r, b)` (in image pixels). This conversion is done with the
        following equations:

            ```
            l = x_loc - l^*
            t = y_loc - t^*
            r = r^* + x_loc
            b = b^* + y_loc
            ```

        Parameters
        ----------
        boxes_delta:
            Boxes in the delta domain, with shape `(batch_size, 4, y, x)`. Where `x` and
            `y` indicate the indices into the feature map. The four elements in the
            second dimension are `(l^*, t^*, r^*, b^*)`.

        locations:
            The location of each feature map pixel in the original image. This should
            have a shape of `(y*x, 2)`. The second dimension has elements
            `(x_loc, y_loc)`.

        Returns
        -------
        boxes_image:
            Boxes in the image domain, with shape `(batch_size, 4, y, x)`. Where `x` and
            `y` indicate the indices into the feature map. The four elements in the
            second dimension are `(l, t, r, b)`.
        """
        # Convert the locations tensor to the same shape as the box deltas
        batch_size = boxes_delta.shape[0]
        height = boxes_delta.shape[2]
        width = boxes_delta.shape[3]
        locations = locations.reshape((height, width, 2))
        locations = locations.unsqueeze(0)  # Add a batch dim
        locations = locations.repeat(batch_size, 1, 1, 1)  # Populate batch
        locations = locations.permute(0, 3, 1, 2)  # (B, H, W, L) ->  (B, L, H, W)

        # Convert to image domain
        box_image = boxes_delta
        box_image[:, 0, :, :] = locations[:, 0, :, :] - box_image[:, 0, :, :]  # left
        box_image[:, 1, :, :] = locations[:, 1, :, :] - box_image[:, 1, :, :]  # top
        box_image[:, 2, :, :] = locations[:, 0, :, :] + box_image[:, 2, :, :]  # right
        box_image[:, 3, :, :] = locations[:, 1, :, :] + box_image[:, 3, :, :]  # bottom

        return box_image

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
    def calculate_targets(
        feature_map: FeatureMap, objects: FrameLabels
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the class ID and box regression targets for a specific feature map.
        This determines which objects correspond to the feature map, then it allocates
        the correct class ID and box regression target to each pixel in the feature map.

        Parameters
        ----------
        feature_map:
            The feature map we want to find objects for.

        objects:
            All objects associated with the frame.

        Returns
        -------
        class_id_targets:

        box_delta_targets:

        box_image_targets
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

        # Calculate the size of each box - this is used to determine the order of
        # presidence. That is, if two boxes are competing for the same feature map pixel
        # then the box with the smaller area gets the pixel
        box_areas = box_area(object_boxes_xyxy)

        # Descending sort boxes and class ID by box area
        indices = torch.sort(box_areas, descending=True).indices
        object_boxes_xyxy = object_boxes_xyxy[indices, :]
        class_ids = class_ids[indices]

        num_locations = image_locations_xy.shape[0]

        # Loop through all ground truth objects and determine if each pixel in the
        # feature map belongs to the object - additionally calculate the regression
        # targets for the object at every pixel
        class_id_targets = class_ids.new_zeros((num_locations,))
        box_delta_targets = object_boxes_xyxy.new_zeros((num_locations, 4))
        box_image_targets = object_boxes_xyxy.new_zeros((num_locations, 4))
        for object_box_xyxy, class_id in zip(object_boxes_xyxy, class_ids, strict=True):
            # Calculate all regression targets for this box
            dupe_box_xyxy = object_box_xyxy.expand((num_locations, -1))
            # trunk-ignore(ruff/E741)
            l = image_locations_xy[:, 0] - dupe_box_xyxy[:, 0]
            t = image_locations_xy[:, 1] - dupe_box_xyxy[:, 1]
            r = dupe_box_xyxy[:, 2] - image_locations_xy[:, 0]
            b = dupe_box_xyxy[:, 3] - image_locations_xy[:, 1]
            box_deltas = torch.stack([l, t, r, b], dim=1)

            # Determine if any meet the size requirements
            mask = min_width / 2 <= box_deltas[:, ::2].min(dim=1).values
            mask &= box_deltas[:, ::2].max(dim=1).values < max_width / 2
            mask &= min_height / 2 <= box_deltas[:, 1::2].min(dim=1).values
            mask &= box_deltas[:, 1::2].max(dim=1).values < max_height / 2

            # Update the regression targets
            class_id_targets[mask] = class_id
            box_delta_targets[mask, :] = box_deltas[mask, :]
            box_image_targets[mask, :] = dupe_box_xyxy[mask, :]

        return class_id_targets, box_delta_targets, box_image_targets

    @staticmethod
    def batch_loss(
        batch_feature_maps: list[list[FeatureMap]],
        objects: list[FrameLabels],
        batch_logits: Tensor,
        batch_box_predictions: Tensor,
        batch_centerness: Tensor,
    ):
        # Calculate the class and regression targets for each image in the batch
        batch_cls_targets = []
        batch_box_targets = []
        for idx, image_fms in enumerate(batch_feature_maps):
            # Calculate the class and regression targets for each feature map in the
            # image
            image_cls_targets = []
            image_box_targets = []
            for fm in image_fms:
                cls_targs, _, box_targs = FCOSHead.calculate_targets(fm, objects[idx])
                image_cls_targets.append(cls_targs)
                image_box_targets.append(box_targs)

            batch_cls_targets.append(torch.cat(image_cls_targets, dim=0))
            batch_box_targets.append(torch.cat(image_box_targets, dim=0))

        batch_cls_targets = torch.stack(batch_cls_targets, dim=0)
        batch_box_targets = torch.stack(batch_box_targets, dim=0)

        # One hot encode the class targets
        num_classes = batch_logits.shape[-1]
        batch_one_hot_targets = one_hot(batch_cls_targets, num_classes)

        # Calculate the class loss
        cls_loss = sigmoid_focal_loss(
            batch_logits, batch_one_hot_targets, reduction="mean", alpha=-1, gamma=2
        )

        # Calculate the box loss - making sure to only calculate the loss on boxes that
        # are associated with a class
        box_loss = 0
        for im_cls_targets, im_box_targets, im_box_predictions in zip(
            batch_cls_targets, batch_box_targets, batch_box_predictions, strict=True
        ):
            mask = im_cls_targets != 0
            box_targets = im_box_targets[mask, :]
            box_predictions = im_box_predictions[mask, :]
            box_loss += (
                complete_box_iou_loss(box_predictions, box_targets, reduction="sum")
                / mask.sum()
            )

        return cls_loss, box_loss

    @staticmethod
    def loss(
        logits: Tensor,
        box_regs: Tensor,
        centerness: Tensor,
        class_targets: Tensor,
        regression_targets: Tensor,
    ):
        pass
