import json
from pathlib import Path

from ssdv2.structs.exceptions import DatasetError


class DatasetManager:
    """
    Manages the directory structure of a dataset.
    """

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir

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

    def verify_dataset_structure(self):
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

    def create_dataset(self, class_names: dict[int, str]):
        """
        Creates the folders and the `classes.json` file for the dataset.

        Paramaters
        ----------
        class_names:
            A map of class ID to class name. The contents of this is saved to the
            `classes.json` file.
        """

        # Root level folder structure
        self.dataset_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir()
        self.labels_dir.mkdir()

        # Images folder structure
        self.train_images_dir.mkdir()
        self.val_images_dir.mkdir()

        # Labels folder structure
        self.train_labels_dir.mkdir()
        self.val_labels_dir.mkdir()

        # Create class ID to name mapping file
        with open(self.class_file, "w") as fp:
            json.dump(class_names, fp)
