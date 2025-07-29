import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import numpy as np
import pytest
import torch

from ssdv2.dataset import DatasetManager, DatasetSampler
from ssdv2.structs import DataSubset, FrameLabels
from ssdv2.structs.exceptions import DatasetError


class TestDatasetManager:
    @pytest.fixture()
    def valid_dataset_dir(self):
        with TemporaryDirectory() as dir:
            temp_dir = Path(dir)

            # Make the classes file
            classes_file = temp_dir / "classes.json"
            num_classes = 10
            class_names = {idx: f"{uuid4()}" for idx in range(num_classes)}
            with open(classes_file, "w") as fp:
                json.dump(class_names, fp)

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

        # Check the class names have been read correctly
        for key, value in manager.raw_class_names.items():
            assert isinstance(key, int)
            assert isinstance(value, str)

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

    def test_create_train_sampler(self, valid_dataset_dir: Path):
        """
        Test we can create a train data sampler.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        manager = DatasetManager(valid_dataset_dir)
        sampler = manager.create_sampler(DataSubset.TRAIN, dtype, device)
        assert isinstance(sampler, DatasetSampler)
        assert sampler.images_path == manager.train_images_dir
        assert sampler.labels_path == manager.train_labels_dir

    def test_create_val_sampler(self, valid_dataset_dir: Path):
        """
        Test we can create a validation data sampler.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        manager = DatasetManager(valid_dataset_dir)
        sampler = manager.create_sampler(DataSubset.VAL, dtype, device)
        assert isinstance(sampler, DatasetSampler)
        assert sampler.images_path == manager.val_images_dir
        assert sampler.labels_path == manager.val_labels_dir

    def test_add_image_label_pair_valid_image(self, valid_dataset_dir: Path):
        """
        Test that we can add a image and label pair to the dataset.
        """
        dtype = torch.float32
        device = torch.device("cpu")

        with TemporaryDirectory() as temp_dir:
            # Create dummy image
            image_src = Path(temp_dir) / f"{uuid4()}.png"
            image_data = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.imwrite(image_src.as_posix(), image_data)

            # Create dummy label
            boxes = torch.rand((10, 4), dtype=dtype, device=device)
            class_ids = torch.zeros((10,), dtype=torch.int, device=device)
            class_names = {0: "person"}
            objects = FrameLabels(
                boxes=boxes, raw_class_ids=class_ids, raw_class_names=class_names
            )

            manager = DatasetManager(valid_dataset_dir)
            img_dst, lab_dst = manager.add_image_label_pair(
                image_src, objects, DataSubset.TRAIN
            )

            assert img_dst.is_file()
            assert img_dst == valid_dataset_dir / f"images/train/{image_src.name}"
            assert lab_dst.is_file()
            assert lab_dst == valid_dataset_dir / f"labels/train/{image_src.stem}.txt"

    def test_subset_class_names(self, valid_dataset_dir: Path):
        """
        Test that we can correctly subset class names.
        """
        class_ids_subset = [2, 0, 4, 5, 6]
        manager = DatasetManager(valid_dataset_dir)
        class_names = manager.subset_class_names(class_ids_subset)

        # Check the class names are correct
        assert isinstance(class_names, dict)
        assert list(class_names.keys()) == list(range(len(class_ids_subset)))
        expected_names = [manager.raw_class_names[id] for id in class_ids_subset]
        assert list(class_names.values()) == expected_names
