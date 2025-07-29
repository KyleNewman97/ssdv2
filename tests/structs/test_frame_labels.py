from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
import torch

from ssdv2.structs import FrameLabels


class TestFrameLabels:
    @pytest.fixture
    def raw_class_names(self) -> dict[int, str]:
        return {id: f"{uuid4()}" for id in range(10)}

    @staticmethod
    def _create_label_file(dir: Path, contents: str) -> Path:
        label_file = dir / "label.txt"
        with open(label_file, "w") as fp:
            fp.write(contents)

        return label_file

    def test_init(self, raw_class_names: dict[int, str]):
        """
        Test we can initialise a `FrameLabels` object.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((10, 4), dtype=dtype, device=device)
        class_ids = torch.randint(0, 10, (10,), dtype=torch.int, device=device)

        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        assert isinstance(objects, FrameLabels)
        assert objects.boxes.allclose(boxes)
        assert objects.raw_class_ids.equal(class_ids)

    def test_object_class_names(self):
        """
        Test we can correctly list the classes in the image.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((3, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([0, 1, 0], dtype=torch.int, device=device)
        class_names = {0: "person", 1: "dog", 2: "cat"}

        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=class_names
        )

        assert isinstance(objects.object_class_names, list)
        assert len(objects.object_class_names) == class_ids.shape[0]
        assert objects.object_class_names == ["person", "dog", "person"]

    def test_from_file_valid_file(self, raw_class_names: dict[int, str]):
        """
        Test we can create a `FrameLabels` object from a valid label file.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            contents = "0 0.1 0.1 0.05 0.05\n1 0.2 0.2 0.06 0.06\n"
            label_file = self._create_label_file(temp_path, contents)

            objects = FrameLabels.from_file(label_file, raw_class_names, dtype, device)

        assert isinstance(objects, FrameLabels)
        assert objects.boxes.allclose(
            torch.tensor(
                [[0.1, 0.1, 0.05, 0.05], [0.2, 0.2, 0.06, 0.06]],
                dtype=dtype,
                device=device,
            )
        )
        assert objects.raw_class_ids.equal(
            torch.tensor([0, 1], dtype=torch.int, device=device)
        )

    def test_from_file_insufficient_elements(self, raw_class_names: dict[int, str]):
        """
        Test that a ValueError is thrown if the structure of the label file is
        incorrect.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            contents = "0 0.1 0.1 0.05\n1 0.2 0.2 0.06 0.06\n"
            label_file = self._create_label_file(temp_path, contents)

            with pytest.raises(ValueError):
                FrameLabels.from_file(label_file, raw_class_names, dtype, device)

    def test_from_file_invalid_class_id(self, raw_class_names: dict[int, str]):
        """
        Test that a ValueError is thrown if we use an invalid class ID.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            class_id = max(list(raw_class_names.keys())) + 1
            contents = f"{class_id} 0.1 0.1 0.05 0.05\n1 0.2 0.2 0.06 0.06\n"
            label_file = self._create_label_file(temp_path, contents)

            with pytest.raises(ValueError) as err:
                FrameLabels.from_file(label_file, raw_class_names, dtype, device)

            assert "Invalid class ID" in err.value.args[0]

    def test_from_file_invalid_box_coords(self, raw_class_names: dict[int, str]):
        """
        Test that a ValueError is thrown if we use a box outside the bounds of [0, 1].
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            contents = "0 0.1 0.1 0.05 0.05\n1 0.2 1.1 0.06 0.06\n"
            label_file = self._create_label_file(temp_path, contents)

            with pytest.raises(ValueError) as err:
                FrameLabels.from_file(label_file, raw_class_names, dtype, device)

            assert "Box coords" in err.value.args[0]

    def test_from_file_incorrect_type(self, raw_class_names: dict[int, str]):
        """
        Test that a ValueError is thrown if one of the values is a letter.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            contents = "0 0.1 0.1 0.05 a\n1 0.2 0.2 0.06 0.06\n"
            label_file = self._create_label_file(temp_path, contents)

            with pytest.raises(ValueError):
                FrameLabels.from_file(label_file, raw_class_names, dtype, device)

    def test_to_file(self, raw_class_names: dict[int, str]):
        """
        Test that we can write the objects to a file.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        boxes = torch.rand((10, 4), dtype=dtype, device=device)
        class_ids = torch.randint(0, 10, (10,), dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        with TemporaryDirectory() as temp_dir:
            # Try to write the labels out
            file = Path(temp_dir) / "label.txt"
            objects.to_file(file)
            assert file.is_file()

            # Read the labels in
            read_objects = FrameLabels.from_file(file, raw_class_names, dtype, device)

        assert read_objects.boxes.allclose(objects.boxes)
        assert read_objects.raw_class_ids.equal(objects.raw_class_ids)

    def test_contains_true(self, raw_class_names: dict[int, str]):
        """
        Test we can correctly identify when a frame contains an object of that class.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        class_id = 0
        boxes = torch.rand((1, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([class_id], dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        assert objects.contains_class(class_id)

    def test_contains_false(self, raw_class_names: dict[int, str]):
        """
        Test we can correctly identify when a frame does not contain an object of that
        class.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        class_id = 0
        boxes = torch.rand((1, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([class_id], dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        assert not objects.contains_class(class_id + 1)

    def test_change_classes_keep_class(self, raw_class_names: dict[int, str]):
        """
        Test that we keep classes we want to keep.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((2, 4), dtype=dtype, device=device)
        class_ids_list = [0, 1]
        class_ids = torch.tensor(class_ids_list, dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        filtered_objects = objects.change_classes(class_ids)

        assert isinstance(filtered_objects, FrameLabels)
        assert filtered_objects.boxes.allclose(boxes)
        assert filtered_objects.raw_class_ids.equal(class_ids)

        assert list(filtered_objects.raw_class_names.keys()) == class_ids_list
        expected_class_names = [raw_class_names[id] for id in class_ids_list]
        assert list(filtered_objects.raw_class_names.values()) == expected_class_names

    def test_change_classes_discard_class(self, raw_class_names: dict[int, str]):
        """
        Test that we discard the classes we don't want.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((2, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([0, 1], dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        keep_classes = class_ids + (class_ids.max().item() + 1)
        filtered_objects = objects.change_classes(keep_classes)

        assert isinstance(filtered_objects, FrameLabels)
        assert filtered_objects.boxes.shape == (0, 4)
        assert filtered_objects.raw_class_ids.shape == (0,)

        expected_class_names = {
            idx: raw_class_names[id] for idx, id in enumerate(keep_classes.tolist())
        }
        assert filtered_objects.raw_class_names == expected_class_names

    def test_change_classes_swap_class_ids(self, raw_class_names: dict[int, str]):
        """
        Test that we can swap class IDs when re-assigning them.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((2, 4), dtype=dtype, device=device)
        class_ids_list = [1, 0]
        class_ids = torch.tensor(class_ids_list, dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        filtered_objects = objects.change_classes(class_ids)

        assert isinstance(filtered_objects, FrameLabels)
        assert filtered_objects.boxes.allclose(objects.boxes)
        expected_classes = torch.tensor([0, 1], dtype=dtype, device=device)
        assert filtered_objects.raw_class_ids.equal(expected_classes)

        expected_class_names = {
            idx: raw_class_names[id] for idx, id in enumerate(class_ids_list)
        }
        assert filtered_objects.raw_class_names == expected_class_names

    def test_len(self, raw_class_names: dict[int, str]):
        """
        Test we correctly get the number of objects.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((2, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([0, 1], dtype=torch.int, device=device)
        objects = FrameLabels(
            boxes=boxes, raw_class_ids=class_ids, raw_class_names=raw_class_names
        )

        assert len(objects) == 2
