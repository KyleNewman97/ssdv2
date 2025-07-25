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

    def test_init(self):
        """
        Test we can initialise a dataset manager.
        """
        manager = DatasetManager(Path("tests_dir"))
        assert isinstance(manager, DatasetManager)

    def test_verify_dataset_structure_valid(self, valid_dataset_dir: Path):
        """
        Tests correct dataset structures raise no errors.
        """
        manager = DatasetManager(valid_dataset_dir)
        manager.verify_dataset_structure()

    def test_verify_dataset_structure_invalid(self, invalid_dataset_dir: Path):
        """
        Test we get a DatasetError when checking an invalid dataset.
        """
        manager = DatasetManager(invalid_dataset_dir)
        with pytest.raises(DatasetError) as err:
            manager.verify_dataset_structure()
        assert "Images dir:" in err.value.args[0]

    def test_create_dataset(self):
        """
        Test that we can make the dataset folder structure correctly.
        """
        with TemporaryDirectory() as dir:
            temp_dir = Path(dir)
            manager = DatasetManager(temp_dir)

            # Try to create the dataset
            manager.create_dataset({0: "person", 1: "car"})

            # Ensure it was made correctly
            manager.verify_dataset_structure()

    def test_create_dataset_existing_dataset(self, valid_dataset_dir: Path):
        """
        Test that we throw an error when trying to create a dataset in a folder that
        already contains a dataset.
        """
        manager = DatasetManager(valid_dataset_dir)
        with pytest.raises(FileExistsError):
            manager.create_dataset({0: "person"})
