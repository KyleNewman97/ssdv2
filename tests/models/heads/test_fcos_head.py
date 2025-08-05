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

    def test_calculate_targets_no_matches(self):
        """
        Test we can calculate targets when no objects should be assoicated with this
        feature map.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        size = 20
        fm_data = torch.rand((1, 384, size, size), dtype=dtype, device=device)
        fm = FeatureMap(data=fm_data, stride=32, index=0, all_strides=[32, 16, 8, 4])
        objects = FrameLabels(
            boxes=torch.tensor([[0.2, 0.2, 0.4, 0.4]], dtype=dtype, device=device),
            raw_class_ids=torch.tensor([0], dtype=torch.int, device=device),
            raw_class_names={0: "person"},
        )

        class_ids, regression_targets = FCOSHead.calculate_targets(fm, objects)

        assert isinstance(class_ids, Tensor)
        expected_shape = (size * size,)
        assert class_ids.shape == expected_shape
        expected_class_ids = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert class_ids.equal(expected_class_ids)

        assert isinstance(regression_targets, Tensor)
        expected_shape = (size * size, 4)
        assert regression_targets.shape == expected_shape
        expected_targets = torch.zeros(expected_shape, dtype=dtype, device=device)
        assert regression_targets.allclose(expected_targets)

    def test_calculate_targets_with_match(self):
        """
        Test we can calculate targets when an object should be assoicated with the
        feature map.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        size = 20
        fm_data = torch.rand((1, 384, size, size), dtype=dtype, device=device)
        fm = FeatureMap(data=fm_data, stride=32, index=0, all_strides=[32, 16, 8, 4])
        objects = FrameLabels(
            boxes=torch.tensor([[0.4, 0.4, 0.6, 0.6]], dtype=dtype, device=device),
            raw_class_ids=torch.tensor([0], dtype=torch.int, device=device),
            raw_class_names={0: "person"},
        )

        class_ids, regression_targets = FCOSHead.calculate_targets(fm, objects)

        assert isinstance(class_ids, Tensor)
        expected_shape = (size * size,)
        assert class_ids.shape == expected_shape
        expected_class_ids = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert class_ids.equal(expected_class_ids)

        assert isinstance(regression_targets, Tensor)
        expected_shape = (size * size, 4)
        assert regression_targets.shape == expected_shape
        expected_targets = torch.zeros(expected_shape, dtype=dtype, device=device)
        assert regression_targets.allclose(expected_targets)
