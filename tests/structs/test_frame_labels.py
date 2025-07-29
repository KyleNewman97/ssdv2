from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from ssdv2.structs import FrameLabels


class TestFrameLabels:
    @staticmethod
    def _create_label_file(dir: Path, contents: str) -> Path:
        label_file = dir / "label.txt"
        with open(label_file, "w") as fp:
            fp.write(contents)

        return label_file

    def test_init(self):
        """
        Test we can initialise a `FrameLabels` object.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        boxes = torch.rand((10, 4), dtype=dtype, device=device)
        class_ids = torch.randint(0, 10, (10,), dtype=torch.int, device=device)

        objects = FrameLabels(boxes=boxes, raw_class_ids=class_ids)

        assert isinstance(objects, FrameLabels)
        assert objects.boxes.allclose(boxes)
        assert objects.raw_class_ids.equal(class_ids)

    def test_from_file_valid_file(self):
        """
        Test we can create a `FrameLabels` object from a valid label file.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            contents = "0 0.1 0.1 0.05 0.05\n1 0.2 0.2 0.06 0.06\n"
            label_file = self._create_label_file(temp_path, contents)

            objects = FrameLabels.from_file(label_file, dtype, device)

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

    def test_from_file_insufficient_elements(self):
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
                FrameLabels.from_file(label_file, dtype, device)

    def test_from_file_incorrect_type(self):
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
                FrameLabels.from_file(label_file, dtype, device)

    def test_to_file(self):
        """
        Test that we can write the objects to a file.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        boxes = torch.rand((10, 4), dtype=dtype, device=device)
        class_ids = torch.randint(0, 10, (10,), dtype=torch.int, device=device)
        objects = FrameLabels(boxes=boxes, raw_class_ids=class_ids)

        with TemporaryDirectory() as temp_dir:
            # Try to write the labels out
            file = Path(temp_dir) / "label.txt"
            objects.to_file(file)
            assert file.is_file()

            # Read the labels in
            read_objects = FrameLabels.from_file(file, dtype, device)

        assert read_objects.boxes.allclose(objects.boxes)
        assert read_objects.raw_class_ids.equal(objects.raw_class_ids)

    def test_contains_true(self):
        """
        Test we can correctly identify when a frame contains an object of that class.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        class_id = 0
        boxes = torch.rand((1, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([class_id], dtype=torch.int, device=device)
        objects = FrameLabels(boxes=boxes, raw_class_ids=class_ids)

        assert objects.contains_class(class_id)

    def test_contains_false(self):
        """
        Test we can correctly identify when a frame does not contain an object of that
        class.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        class_id = 0
        boxes = torch.rand((1, 4), dtype=dtype, device=device)
        class_ids = torch.tensor([class_id], dtype=torch.int, device=device)
        objects = FrameLabels(boxes=boxes, raw_class_ids=class_ids)

        assert not objects.contains_class(class_id + 1)
