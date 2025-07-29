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
            classes_file = temp_dir / "classes.json"
            with open(classes_file, "w") as fp:
                fp.write('{"0": "person"}')

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
            classes_file = temp_dir / "classes.json"
            with open(classes_file, "w") as fp:
                fp.write('{"0": "person"}')

            # Make the labels dirs
            labels_dir = temp_dir / "labels"
            labels_dir.mkdir()

            yield temp_dir

    def test_init_valid_dataset(self, valid_dataset_dir: Path):
        """
        Tests correct dataset structures raise no errors.
        """
        manager = DatasetManager(valid_dataset_dir)
        assert isinstance(manager, DatasetManager)

    def test_init_invalid_dataset(self, invalid_dataset_dir: Path):
        """
        Test we get a DatasetError when checking an invalid dataset.
        """
        with pytest.raises(DatasetError) as err:
            DatasetManager(invalid_dataset_dir)
        assert "Images dir:" in err.value.args[0]

    def test_create_new_dataset(self):
        """
        Test that we can make the dataset folder structure correctly.
        """
        with TemporaryDirectory() as dir:
            root_dir = Path(dir)
            class_names = {0: "person", 1: "car"}

            # Try to create the dataset
            manager = DatasetManager.create_new_dataset(root_dir, class_names)

        assert isinstance(manager, DatasetManager)

    def test_create_new_dataset_existing_dataset(self, valid_dataset_dir: Path):
        """
        Test that we throw an error when trying to create a dataset in a folder that
        already contains a dataset.
        """
        class_names = {0: "person"}
        with pytest.raises(FileExistsError):
            DatasetManager.create_new_dataset(valid_dataset_dir, class_names)
