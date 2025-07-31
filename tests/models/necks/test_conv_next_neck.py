import torch

from ssdv2.models.necks import ConvNeXtNeck


class TestConvNeXtNeck:
    def test_init(self):
        """
        Test we can initialise the ConvNeXtNeck.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        neck = ConvNeXtNeck(dtype, device)
        assert isinstance(neck, ConvNeXtNeck)

    def test_forward(self):
        """
        Test we can run inference through the neck.
        """

        dtype = torch.float32
        device = torch.device("cpu")

        # Create a dummy input
        data = [
            torch.rand((1, 8, 160, 160), dtype=dtype, device=device),
            torch.rand((1, 16, 80, 80), dtype=dtype, device=device),
        ]
        dims = [sum([d.shape[1] for d in data[:i]]) for i in range(2, len(data) + 1)]

        neck = ConvNeXtNeck(dtype, device, depths=[3] * len(dims), dims=dims)
        out = neck.forward(data)

        # Check the output type and number of feature maps
        assert isinstance(out, list)
        assert len(out) == len(data)

        # The first output feature map should be the same as the last input feature map
        assert out[0].shape == data[-1].shape
        assert out[0].allclose(data[-1])

        # Check the same of all other output feature maps
        for idx in range(1, len(out)):
            expected = data[-(idx + 1)].shape
            expected = (expected[0], dims[idx - 1], expected[2], expected[3])
            assert out[idx].shape == expected
