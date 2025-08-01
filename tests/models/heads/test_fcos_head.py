import torch

from ssdv2.models.heads import FCOSHead


class TestFCOSHead:
    def test_init(self):
        """
        Test that we can initialise the FCOSHead.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        head = FCOSHead(80, 64, 2).to(dtype=dtype, device=device)
        assert isinstance(head, FCOSHead)

    def test_forward(self):
        """
        Test we can run forward inference through the head.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        fms = torch.rand((1, 64, 20, 20), dtype=dtype, device=device)

        head = FCOSHead(80, 64, 2).to(dtype=dtype, device=device)
        logits, box_reg, centerness = head.forward(fms)

        assert isinstance(logits, list)
        assert len(logits) == 1
        assert isinstance(box_reg, list)
        assert len(box_reg) == 1
        assert isinstance(centerness, list)
        assert len(centerness) == 1
