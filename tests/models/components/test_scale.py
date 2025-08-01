import torch
from torch import Tensor

from ssdv2.models.components import Scale


class TestScale:
    def test_init(self):
        """
        Test we can initialise the Scale module.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        model = Scale(initial_value=1).to(dtype=dtype, device=device)
        assert isinstance(model, Scale)
        assert sum([p.numel() for p in model.parameters()]) == 1

    def test_forward(self):
        """
        Test we can run inference through Scale.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        initial_value = 2
        model = Scale(initial_value=initial_value).to(dtype=dtype, device=device)

        # Create dummy data
        x = torch.rand((1, 3, 20, 20), dtype=dtype, device=device)

        out = model.forward(x)

        assert isinstance(out, Tensor)
        assert out.shape == x.shape
        assert out.allclose(x * initial_value)
