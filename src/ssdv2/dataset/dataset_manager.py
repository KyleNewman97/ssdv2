import json
from pathlib import Path

from ssdv2.structs.exceptions import DatasetError


class DatasetManager:
    """
    Manages the directory structure of a dataset.
    """

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
            class_names: dict[int, str] = json.load(fp)

        self.dataset_dir = dataset_dir
        self.class_names = class_names
        self.verify_dataset_structure()

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
