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
        fm_channels = [fm.data.shape[1] for fm in fms]

        neck = ConvNeXtNeck(depths=[0, 3], fm_channels=fm_channels)
        neck = neck.to(dtype=dtype, device=device)
        out = neck.forward(fms)

        # Check the output type and number of feature maps
        assert isinstance(out, list)
        assert len(out) == len(fms)

        # Check all feature maps
        for idx in range(len(out)):
            shape = fms[-(idx + 1)].data.shape
            shape = (shape[0], neck.out_fm_channels[idx], shape[2], shape[3])
            assert out[idx].data.shape == shape
            assert out[idx].stride == fms[-(idx + 1)].stride
            assert out[idx].index == idx
            assert out[idx].all_strides == fms[-(idx + 1)].all_strides[::-1]
