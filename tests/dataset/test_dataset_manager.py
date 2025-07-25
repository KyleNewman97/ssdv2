from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ssdv2.dataset import DatasetManager
from ssdv2.structs.exceptions import DatasetError


class TestDatasetManager:
    @pytest.fixture()
    def valid_dataset_dir(self):
        with TemporaryDirectory() as dir:
            temp_dir = Path(dir)

            # Make the classes file
            (temp_dir / "classes.json").touch()

            # Make images and labels dirs
            images_dir = temp_dir / "images"
            labels_dir = temp_dir / "labels"
            images_dir.mkdir()
            labels_dir.mkdir()

            # Make image dirs
            train_images_dir = images_dir / "train"
            val_images_dir = images_dir / "val"
            train_images_dir.mkdir()
            val_images_dir.mkdir()

            # Make label dirs
            train_labels_dir = labels_dir / "train"
            val_labels_dir = labels_dir / "val"
            train_labels_dir.mkdir()
            val_labels_dir.mkdir()

            yield temp_dir

    @pytest.fixture()
    def invalid_dataset_dir(self):
        with TemporaryDirectory() as dir:
            temp_dir = Path(dir)

            # Make the classes file
            (temp_dir / "classes.json").touch()

            # Make the labels dirs
            labels_dir = temp_dir / "labels"
            labels_dir.mkdir()

            yield temp_dir

    def test_init_valid_dataset(self, valid_dataset_dir: Path):
        """
        Tests we can initialise a dataset that is structured correctly.
        """
        manager = DatasetManager(valid_dataset_dir)
        assert isinstance(manager, DatasetManager)

    def test_init_invalid_dataset(self, invalid_dataset_dir: Path):
        """
        Test we get a DatasetError when initialising on an invalid dataset dir.
        """
        with pytest.raises(DatasetError) as err:
            DatasetManager(invalid_dataset_dir)
        assert "Images dir:" in err.value.args[0]
