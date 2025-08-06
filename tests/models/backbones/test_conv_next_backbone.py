import math

import torch

from ssdv2.models.backbones import ConvNeXtBackbone
from ssdv2.structs import FeatureMap


class TestConvNeXtBackbone:
    def test_init(self):
        """
        Test we can initialise the backbone.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        backbone = ConvNeXtBackbone().to(dtype=dtype, device=device)
        assert isinstance(backbone, ConvNeXtBackbone)

    def test_forward(self):
        """
        Test the forward function of the backbone.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        depths = [3, 3]
        dims = [48, 96]
        strides = [4, 2]
        backbone = ConvNeXtBackbone(depths, dims, strides)
        backbone = backbone.to(dtype=dtype, device=device)

        # Create dummy data
        img = torch.rand((1, 3, 640, 640), dtype=dtype, device=device)

        # Run the forward operation
        fms = backbone.forward(img)

        assert isinstance(fms, list)
        assert len(fms) == backbone.num_stages
        expected_strides: list[int] = []
        for idx, fm in enumerate(fms):
            assert isinstance(fm, FeatureMap)
            expected_stride = math.prod(strides[: (idx + 1)])
            assert fm.data.shape == (
                1,
                dims[idx],
                img.shape[2] // expected_stride,
                img.shape[3] // expected_stride,
            )
            assert fm.stride == expected_stride
            assert fm.index == idx
            expected_strides.append(fm.stride)
        for fm in fms:
            assert fm.all_strides == expected_strides
