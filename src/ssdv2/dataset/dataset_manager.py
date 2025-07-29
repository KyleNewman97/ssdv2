import json
from pathlib import Path

import torch

from ssdv2.dataset.dataset_sampler import DatasetSampler
from ssdv2.structs import DataSubset, FrameLabels
from ssdv2.structs.exceptions import DatasetError


class DatasetManager:
    """
    Manages the directory structure of a dataset.
    """

    @property
    def images_dir(self) -> Path:
        return self.dataset_dir / "images"

    @property
    def labels_dir(self) -> Path:
        return self.dataset_dir / "labels"

    @property
    def train_images_dir(self) -> Path:
        return self.images_dir / "train"

    @property
    def val_images_dir(self) -> Path:
        return self.images_dir / "val"

    @property
    def train_labels_dir(self) -> Path:
        return self.labels_dir / "train"

    @property
    def val_labels_dir(self) -> Path:
        return self.labels_dir / "val"

    @property
    def class_file(self) -> Path:
        return self.dataset_dir / "classes.json"

    def __init__(self, dataset_dir: Path):
        """
        Initialise a DatasetManager for an existing dataset.

        Parameters
        ----------
        dataset_dir:
            The root folder of the dataset.
        """
        # Check that the classes file exists
        classes_file = dataset_dir / "classes.json"
        if not classes_file.is_file():
            raise DatasetError(f"No classes file at {classes_file}.")

        # Read in the contents of the classes file
        with open(classes_file, "r") as fp:
            contents: dict[str, str] = json.load(fp)
            raw_class_names: dict[int, str] = {
                int(id): name for id, name in contents.items()
            }

        self.dataset_dir = dataset_dir
        self.raw_class_names = raw_class_names
        self._verify_dataset_structure()

    @classmethod
    def create_new_dataset(
        cls, dataset_dir: Path, class_names: dict[int, str]
    ) -> "DatasetManager":
        """
        Creates the folders and the `classes.json` file for the dataset.

        Paramaters
        ----------
        class_names:
            A map of class ID to class name. The contents of this is saved to the
            `classes.json` file.
        """

        # Root level folder structure
        dataset_dir.mkdir(exist_ok=True)
        (dataset_dir / "images").mkdir()
        (dataset_dir / "labels").mkdir()

        # Images folder structure
        (dataset_dir / "images/train").mkdir()
        (dataset_dir / "images/val").mkdir()

        # Labels folder structure
        (dataset_dir / "labels/train").mkdir()
        (dataset_dir / "labels/val").mkdir()

        # Create class ID to name mapping file
        with open(dataset_dir / "classes.json", "w") as fp:
            json.dump(class_names, fp)

        return cls(dataset_dir)

    def _verify_dataset_structure(self):
        """
        Verify that the directory structure is correct.
        """
        if not self.class_file.is_file():
            raise DatasetError(f"Classes file: {self.class_file} not a file.")

        if not self.dataset_dir.is_dir():
            raise DatasetError(f"Dataset dir: {self.dataset_dir} not a dir.")

        if not self.images_dir.is_dir():
            raise DatasetError(f"Images dir: {self.images_dir} not a dir.")

        if not self.labels_dir.is_dir():
            raise DatasetError(f"Labels dir: {self.labels_dir} not a dir.")

        if not self.train_images_dir.is_dir():
            raise DatasetError(f"Train images dir: {self.train_images_dir} not a dir.")

        if not self.val_images_dir.is_dir():
            raise DatasetError(f"Val images dir: {self.val_images_dir} not a dir.")

        if not self.train_labels_dir.is_dir():
            raise DatasetError(f"Train labels dir: {self.train_labels_dir} not a dir.")

        if not self.val_labels_dir.is_dir():
            raise DatasetError(f"Val labels dir: {self.val_labels_dir} not a dir.")

    def create_sampler(
        self, subset: DataSubset, dtype: torch.dtype, device: torch.device
    ) -> DatasetSampler:
        """
        Create a dataset sampler for the specified subset.

        Parameters
        ----------
        subset:
            The subset of the dataset to create the sampler for.

        dtype:
            What format the underlying data will be loaded in with.

        device:
            The device the data will be loaded on to.
        """
        if subset == DataSubset.TRAIN:
            return DatasetSampler(
                self.train_images_dir,
                self.train_labels_dir,
                self.raw_class_names,
                dtype,
                device,
            )
        elif subset == DataSubset.VAL:
            return DatasetSampler(
                self.val_images_dir,
                self.val_labels_dir,
                self.raw_class_names,
                dtype,
                device,
            )
        else:
            raise NotImplementedError(f"{subset} not supported yet.")

    def add_image_label_pair(
        self, image_src: Path, objects: FrameLabels, subset: DataSubset
    ) -> tuple[Path, Path]:
        """
        Adds an image and label pair to the dataset. The stem of the image file is used
        to name to new entry in the dataset.

        Parameters
        ----------
        image_src:
            Path to the image file to place in the new dataset. This is done by creating
            a symlink to this file. This saves both memory and time.

        objects:
            The object labels for the associated image.

        subset:
            The dataset subset to save the new image and label pair to.

        Returns
        -------
        image_dst:
            The location the image symlink is created at.

        label_dst:
            The location the label file is created at.
        """
        if subset == DataSubset.TRAIN:
            image_dst = self.train_images_dir / image_src.name
            label_dst = self.train_labels_dir / (image_src.stem + ".txt")
        elif subset == DataSubset.VAL:
            image_dst = self.val_images_dir / image_src.name
            label_dst = self.val_labels_dir / (image_src.stem + ".txt")
        else:
            raise NotImplementedError(f"{subset} not supported yet.")

        if image_dst.exists():
            raise RuntimeError(f"{image_dst} already exists.")
        if label_dst.exists():
            raise RuntimeError(f"{label_dst} already exists.")

        image_dst.symlink_to(image_src)
        objects.to_file(label_dst)

        return image_dst, label_dst

    def subset_class_names(self, class_ids_subset: list[int]) -> dict[int, str]:
        """
        Creates a subset of the class names based on the provided class IDs. When the
        class names are extracted they are given a new class ID so they remain
        sequential.

        Parameters
        ----------
        class_ids_subset:
            The subset of class IDs we want to keep.

        Returns
        -------
        class_names_subset:
            The subset of class names contained within the `class_ids_subset`. The class
            IDs are re-numbered to ensure they are sequential.
        """
        # Create a map from the new class IDs to the class names
        class_names_subset = {
            idx: self.raw_class_names[id] for idx, id in enumerate(class_ids_subset)
        }

        return class_names_subset
