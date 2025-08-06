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

        head = FCOSHead(80, [64]).to(dtype=dtype, device=device)
        assert isinstance(head, FCOSHead)

    def test_forward(self):
        """
        Test we can run forward inference through the head.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        bs = 4
        num_cls = 80
        data0 = torch.rand((bs, 192, 80, 80), dtype=dtype, device=device)
        data1 = torch.rand((bs, 96, 160, 160), dtype=dtype, device=device)
        fms = [
            FeatureMap(data=data0, stride=4, index=0, all_strides=[4, 8]),
            FeatureMap(data=data1, stride=8, index=1, all_strides=[4, 8]),
        ]

        in_channels = [fm.data.shape[1] for fm in fms]
        head = FCOSHead(num_cls, in_channels).to(dtype=dtype, device=device)
        logits, centerness, box_reg = head.forward(fms)

        num_fm_pixels = sum([fm.height * fm.width for fm in fms])
        assert isinstance(logits, Tensor)
        assert logits.shape == (bs, num_fm_pixels, num_cls)
        assert isinstance(centerness, Tensor)
        assert centerness.shape == (bs, num_fm_pixels, 1)
        assert isinstance(box_reg, Tensor)
        assert box_reg.shape == (bs, num_fm_pixels, 4)

    def test_delta_to_image_domain(self):
        """
        Test we can convert from delta to image domain.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Perpare boxes in the delta domain
        boxes_delta = torch.tensor([[[100, 200, 300, 240]]], dtype=dtype, device=device)
        boxes_delta = boxes_delta.unsqueeze(0).repeat(8, 1, 1, 1)
        boxes_delta = boxes_delta.permute(0, 3, 1, 2)

        # Prepare locations
        locations = torch.tensor([[320, 320]], dtype=dtype, device=device)

        boxes_image = FCOSHead.delta_to_image_domain(boxes_delta, locations)

        # Check type and shape
        assert isinstance(boxes_image, Tensor)
        assert boxes_image.shape == boxes_delta.shape

        # Check values
        expected = torch.tensor([[[220, 120, 620, 560]]], dtype=dtype, device=device)
        expected = expected.unsqueeze(0).repeat(8, 1, 1, 1)
        expected = expected.permute(0, 3, 1, 2)
        assert boxes_image.allclose(expected)

    def test_delta_to_image_domain_representative(self):
        """
        Test we can convert from delta to image domain with representative sized inputs.
        This does not check the calculation was done correctly.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        boxes_delta = torch.rand((8, 4, 20, 20), dtype=dtype, device=device)
        locations = torch.rand((400, 2), dtype=dtype, device=device)

        boxes_image = FCOSHead.delta_to_image_domain(boxes_delta, locations)

        assert isinstance(boxes_image, Tensor)
        assert boxes_image.shape == boxes_delta.shape

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

        class_ids, delta_targs, image_targs = FCOSHead.calculate_targets(fm, objects)

        assert isinstance(class_ids, Tensor)
        expected_shape = (size * size,)
        assert class_ids.shape == expected_shape
        expected_class_ids = torch.zeros(expected_shape, dtype=torch.int, device=device)
        assert class_ids.equal(expected_class_ids)

        assert isinstance(delta_targs, Tensor)
        expected_shape = (size * size, 4)
        assert delta_targs.shape == expected_shape
        expected_targets = torch.zeros(expected_shape, dtype=dtype, device=device)
        assert delta_targs.allclose(expected_targets)

        assert isinstance(image_targs, Tensor)
        expected_shape = (size * size, 4)
        assert image_targs.shape == expected_shape
        expected_targets = torch.zeros(expected_shape, dtype=dtype, device=device)
        assert image_targs.allclose(expected_targets)

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
            boxes=torch.tensor([[0.4, 0.6, 0.6, 0.6]], dtype=dtype, device=device),
            raw_class_ids=torch.tensor([0], dtype=torch.int, device=device),
            raw_class_names={0: "person"},
        )

        class_ids, delta_targs, image_targs = FCOSHead.calculate_targets(fm, objects)

        # Check output type and shape is correct
        assert isinstance(class_ids, Tensor)
        class_id_shape = (size * size,)
        assert class_ids.shape == class_id_shape
        assert isinstance(delta_targs, Tensor)
        box_shape = (size * size, 4)
        assert delta_targs.shape == box_shape
        assert isinstance(image_targs, Tensor)
        assert image_targs.shape == box_shape

        # Check the class ID assignment is correct - this result was calculated by hand
        expected_class_ids = torch.zeros(class_id_shape, dtype=torch.int, device=device)
        expected_class_ids[227] = 1
        expected_class_ids[228] = 1
        expected_class_ids[247] = 1
        expected_class_ids[248] = 1
        assert class_ids.equal(expected_class_ids)

        # Check the box delta targets are correct - this result was calculated by hand
        expected_targets = torch.zeros(box_shape, dtype=dtype, device=device)
        expected_targets[227, :] = expected_targets.new_tensor([176, 176, 208, 208])
        expected_targets[228, :] = expected_targets.new_tensor([208, 176, 176, 208])
        expected_targets[247, :] = expected_targets.new_tensor([176, 208, 208, 176])
        expected_targets[248, :] = expected_targets.new_tensor([208, 208, 176, 176])
        assert delta_targs.allclose(expected_targets)

        # Check the box image targets are correct - this result was calculated by hand
        expected_targets = torch.zeros(box_shape, dtype=dtype, device=device)
        expected_targets[227, :] = expected_targets.new_tensor([64, 192, 448, 576])
        expected_targets[228, :] = expected_targets.new_tensor([64, 192, 448, 576])
        expected_targets[247, :] = expected_targets.new_tensor([64, 192, 448, 576])
        expected_targets[248, :] = expected_targets.new_tensor([64, 192, 448, 576])
        assert image_targs.allclose(expected_targets)
