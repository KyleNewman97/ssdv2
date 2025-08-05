import torch
from torch import Tensor

from ssdv2.models.heads import FCOSHead
from ssdv2.structs import FeatureMap, FrameLabels


class TestFCOSHead:
    def test_init(self):
        """
        Test that we can initialise the FCOSHead.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        head = FCOSHead(80, 64).to(dtype=dtype, device=device)
        assert isinstance(head, FCOSHead)

    def test_forward(self):
        """
        Test we can run forward inference through the head.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        batch_size = 1
        num_classes = 80
        channels = 384
        size = 20
        fm_data = torch.rand(
            (batch_size, channels, size, size), dtype=dtype, device=device
        )
        fm = FeatureMap(data=fm_data, stride=4, index=3, all_strides=[4])

        head = FCOSHead(num_classes, channels).to(dtype=dtype, device=device)
        logits, box_reg, centerness = head.forward(fm)

        assert isinstance(logits, Tensor)
        assert logits.shape == (batch_size, num_classes, size, size)
        assert isinstance(box_reg, Tensor)
        assert box_reg.shape == (batch_size, 4, size, size)
        assert isinstance(centerness, Tensor)
        assert centerness.shape == (batch_size, 1, size, size)

    def test_calculate_feature_map_locations(self):
        """
        Test we can calculate feature map locations.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        fm_data = torch.rand((1, 32, 4, 2), dtype=dtype, device=device)
        fm = FeatureMap(data=fm_data, stride=16, index=3, all_strides=[16])

        indices = FCOSHead.calculate_feature_map_locations(fm)

        assert isinstance(indices, Tensor)
        assert indices.shape == (fm.height * fm.width, 2)

        # Ordered as top-left moving along columns then rows with each element being
        # arranged as (y, x)
        expected_indices = torch.tensor(
            [[8, 8], [24, 8], [8, 24], [24, 24], [8, 40], [24, 40], [8, 56], [24, 56]],
            dtype=dtype,
            device=device,
        )
        assert indices.equal(expected_indices)

    def test_filter_objects_by_feature_map(self):
        """
        Test we can
        """
        dtype = torch.float32
        device = torch.device("cpu")

        fm_data = torch.rand((1, 384, 20, 20), dtype=dtype, device=device)
        fm = FeatureMap(data=fm_data, stride=32, index=0, all_strides=[32, 16, 8, 4])
        objects = FrameLabels(
            boxes=torch.tensor([[0.2, 0.2, 0.4, 0.4]], dtype=dtype, device=device),
            raw_class_ids=torch.tensor([0], dtype=torch.int, device=device),
            raw_class_names={0: "person"},
        )

        FCOSHead.filter_objects_by_feature_map(fm, objects)
