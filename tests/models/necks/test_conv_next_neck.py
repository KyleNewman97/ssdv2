import torch

from ssdv2.models.necks import ConvNeXtNeck
from ssdv2.structs import FeatureMap


class TestConvNeXtNeck:
    def test_init(self):
        """
        Test we can initialise the ConvNeXtNeck.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        neck = ConvNeXtNeck().to(dtype=dtype, device=device)
        assert isinstance(neck, ConvNeXtNeck)

    def test_forward(self):
        """
        Test we can run inference through the neck.
        """

        dtype = torch.float32
        device = torch.device("cpu")

        # Create a dummy input
        fms = [
            FeatureMap(
                data=torch.rand((1, 8, 160, 160), dtype=dtype, device=device),
                stride=4,
                index=0,
                all_strides=[4, 8],
            ),
            FeatureMap(
                data=torch.rand((1, 16, 80, 80), dtype=dtype, device=device),
                stride=8,
                index=1,
                all_strides=[4, 8],
            ),
        ]
        dims = [sum([d.data.shape[1] for d in fms[:i]]) for i in range(2, len(fms) + 1)]

        neck = ConvNeXtNeck(depths=[3] * len(dims), dims=dims)
        neck = neck.to(dtype=dtype, device=device)
        out = neck.forward(fms)

        # Check the output type and number of feature maps
        assert isinstance(out, list)
        assert len(out) == len(fms)

        # The first output feature map should be the same as the last input feature map
        assert out[0].data.shape == fms[-1].data.shape
        assert out[0].data.allclose(fms[-1].data)

        # Check the same of all other output feature maps
        for idx in range(1, len(out)):
            expected = fms[-(idx + 1)].data.shape
            expected = (expected[0], dims[idx - 1], expected[2], expected[3])
            assert out[idx].data.shape == expected
            assert out[idx].stride == fms[-(idx + 1)].stride
            assert out[idx].index == idx
            assert out[idx].all_strides == fms[-(idx + 1)].all_strides[::-1]
