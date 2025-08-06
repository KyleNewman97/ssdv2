import torch

from ssdv2.structs import FeatureMap


class TestFeatureMap:
    def test_init(self):
        """
        Test we can initialise a feature map.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=4, index=0, all_strides=[4])

        assert isinstance(fm, FeatureMap)
        assert fm.width == width
        assert fm.height == height

    def test_image_width_height_no_stride(self):
        """
        Test we can calculate the width and height of the original image when we have no
        stride.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=0, index=0, all_strides=[4])

        assert fm.image_width == width
        assert fm.image_height == height

    def test_image_width_height_with_stride(self):
        """
        Test we can calculate the width and height of the original image when we have a
        stride.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=4, index=0, all_strides=[4])

        assert fm.image_width == width * fm.stride
        assert fm.image_height == height * fm.stride

    def test_fcos_min_object_width_height_min_stride(self):
        """
        Test we can calculate the minimum object size when we are at the minimum feature
        map stride.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=4, index=0, all_strides=[4, 8, 16])

        assert fm.fcos_min_object_width == 0
        assert fm.fcos_min_object_height == 0

    def test_fcos_min_object_width_height_other_stride(self):
        """
        Test we can calculate the minimum object size when we are not at the minimum
        stride.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=8, index=1, all_strides=[4, 8, 16])

        assert fm.fcos_min_object_width == 1 / 4 * fm.image_width
        assert fm.fcos_min_object_height == 1 / 4 * fm.image_height

    def test_fcos_max_object_width_height_max_stride(self):
        """
        Test we can calculate the maximum object size when we are at the maximum feature
        map stride.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=16, index=2, all_strides=[4, 8, 16])

        assert fm.fcos_max_object_width == torch.inf
        assert fm.fcos_max_object_height == torch.inf

    def test_fcos_max_object_width_height_other_stride(self):
        """
        Test we can calculate the maximum object size when we are not at the maximum
        stride.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        height, width = 18, 12
        data = torch.zeros((2, 3, height, width), dtype=dtype, device=device)
        fm = FeatureMap(data=data, stride=8, index=1, all_strides=[4, 8, 16])

        assert fm.fcos_max_object_width == 1 / 2 * fm.image_width
        assert fm.fcos_max_object_height == 1 / 2 * fm.image_height
