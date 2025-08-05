import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class FeatureMap(BaseModel):
    """
    A feature map that can be returned from arbitrary levels in the network.
    """

    data: Tensor = Field(description="Feature map data.")
    stride: int = Field(description="Cumulative stride up to this feature map.")
    index: int = Field(description="Feature map index of those returned by the module.")
    all_strides: list[int] = Field(description="All strides returned by the module.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def width(self) -> int:
        """
        Width of the feature map.
        """
        return self.data.shape[3]

    @property
    def height(self) -> int:
        """
        Height of the feature map.
        """
        return self.data.shape[2]

    @property
    def image_width(self) -> int:
        """
        Width of the original image in pixels.
        """
        return self.width * self.stride if self.stride != 0 else self.width

    @property
    def image_height(self) -> int:
        """
        Height of the original image in pixels.
        """
        return self.height * self.stride if self.stride != 0 else self.height

    @property
    def fcos_min_object_width(self) -> float:
        """
        The minimum object width (in pixels) we should be able to detect from this
        feature map. This is specific to the FCOS head.
        """
        min_stride = min(self.all_strides)
        max_stride = max(self.all_strides)

        if self.stride == min_stride:
            # If this feature map has the minimum cumulative stride then we want to be
            # able to detect the smallest objects possible
            return 0
        else:
            # For any other feature map we can calculate the minimum object size it
            # should support by the ratio of its cumulative stride to the max stride
            # divided by two
            return self.image_width * self.stride / (max_stride * 2)

    @property
    def fcos_min_object_height(self) -> float:
        """
        The minimum object height (in pixels) we should be able to detect from this
        feature map. This is specific to the FCOS head.
        """
        min_stride = min(self.all_strides)
        max_stride = max(self.all_strides)

        if self.stride == min_stride:
            # If this feature map has the minimum cumulative stride then we want to be
            # able to detect the smallest objects possible
            return 0
        else:
            # For any other feature map we can calculate the minimum object size it
            # should support by the ratio of its cumulative stride to the max stride
            # divided by two
            return self.image_height * self.stride / (max_stride * 2)

    @property
    def fcos_max_object_width(self) -> float:
        """
        The maximum object width (in pixels) we should be able to detect from this
        feature map. This is specific to the FCOS head.
        """
        max_stride = max(self.all_strides)

        if self.stride == max_stride:
            # If this feature map has the maximum cumulative stride then we want to be
            # able to detect the largest objects possible
            return torch.inf
        else:
            # For any other feature map we can calculate the minimum object size it
            # should support by the ratio of its cumulative stride to the max stride
            return self.image_width * self.stride / max_stride

    @property
    def fcos_max_object_height(self) -> float:
        """
        The maximum object height (in pixels) we should be able to detect from this
        feature map. This is specific to the FCOS head.
        """
        max_stride = max(self.all_strides)

        if self.stride == max_stride:
            # If this feature map has the maximum cumulative stride then we want to be
            # able to detect the largest objects possible
            return torch.inf
        else:
            # For any other feature map we can calculate the minimum object size it
            # should support by the ratio of its cumulative stride to the max stride
            return self.image_height * self.stride / max_stride
