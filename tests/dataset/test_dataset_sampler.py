from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import pytest
import torch
from PIL import Image

from ssdv2.dataset import DatasetSampler


class TestDatasetSample:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.images_path = self.temp_path / "images"
        self.labels_path = self.temp_path / "labels"

        self.images_path.mkdir(exist_ok=True, parents=True)
        self.labels_path.mkdir(exist_ok=True, parents=True)

        yield

        self.temp_dir.cleanup()

    def test_init_valid_dataset(self):
        """
        Test that we keep valid samples when initialising the dataset.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        uuid = f"{uuid4()}"
        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")
        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1 0.1")
        class_names = {0: "person"}

        # Try to initialise the dataset
        dataset = DatasetSampler(
            self.images_path, self.labels_path, class_names, dtype, device
        )
        assert isinstance(dataset, DatasetSampler)
        assert len(dataset) == 1

    def test_init_missing_label(self):
        """
        Test that we remove samples when the label file is missing.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        uuid = f"{uuid4()}"
        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")
        class_names = {0: "person"}

        # Try to initialise the dataset
        dataset = DatasetSampler(
            self.images_path, self.labels_path, class_names, dtype, device
        )
        assert isinstance(dataset, DatasetSampler)
        assert len(dataset) == 0

    def test_init_invalid_label_num_elements(self):
        """
        Test that we remove samples that contain label rows with not enough elements.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        uuid = f"{uuid4()}"
        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")
        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1")
        class_names = {0: "person"}

        # Try to initialise the dataset
        dataset = DatasetSampler(
            self.images_path, self.labels_path, class_names, dtype, device
        )
        assert isinstance(dataset, DatasetSampler)
        assert len(dataset) == 0

    def test_init_invalid_label_class(self):
        """
        Test that we remove samples that contain an invalid class ID.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        uuid = f"{uuid4()}"
        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")
        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("2 0.1 0.1 0.1 0.1")
        class_names = {0: "person"}

        # Try to initialise the dataset
        dataset = DatasetSampler(
            self.images_path, self.labels_path, class_names, dtype, device
        )
        assert isinstance(dataset, DatasetSampler)
        assert len(dataset) == 0

    def test_init_invalid_label_box(self):
        """
        Test that we remove samples that contain an invalid box.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        # Create dummy data
        uuid = f"{uuid4()}"
        im = Image.fromarray(np.zeros((320, 320, 3), dtype=np.uint8))
        im.save(self.images_path / f"{uuid}.png")
        with open(self.labels_path / f"{uuid}.txt", "w") as fp:
            fp.write("0 0.1 0.1 0.1 -0.1")
        class_names = {0: "person"}

        # Try to initialise the dataset
        dataset = DatasetSampler(
            self.images_path, self.labels_path, class_names, dtype, device
        )
        assert isinstance(dataset, DatasetSampler)
        assert len(dataset) == 0
