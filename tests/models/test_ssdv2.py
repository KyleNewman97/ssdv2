import torch
from torch import Tensor

from ssdv2.models import SSDv2


class TestSSDv2:
    def test_init(self):
        """
        Test we can initialise the model.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        model = SSDv2(80).to(dtype=dtype, device=device)
        assert isinstance(model, SSDv2)

    def test_forward(self):
        """
        Tes we can run forward inference with the model.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        images = torch.rand((2, 3, 640, 640), dtype=dtype, device=device)

        # Run inference
        num_classes = 80
        model = SSDv2(num_classes).to(dtype=dtype, device=device)
        logits, centerness, boxes = model.forward(images)

        assert isinstance(logits, Tensor)
        assert isinstance(centerness, Tensor)
        assert isinstance(boxes, Tensor)
