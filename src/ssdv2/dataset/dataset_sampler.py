from pathlib import Path

import torch
from torch.utils.data import Dataset

from ssdv2.structs import FrameLabels
from ssdv2.utils import MetaLogger


class DatasetSampler(Dataset, MetaLogger):
    """
    Responsible for sampling image and label pairs from a dataset.
    """

    def __init__(
        self,
        images_path: Path,
        labels_path: Path,
        raw_class_names: dict[int, str],
        dtype: torch.dtype,
        device: torch.device,
    ):
        Dataset.__init__(self)
        MetaLogger.__init__(self)

        self.images_path = images_path
        self.labels_path = labels_path
        self.raw_class_names = raw_class_names
        self.device = device
        self.dtype = dtype

        self.logger.info(f"Images path: {images_path}")
        self.logger.info(f"Labels path: {labels_path}")

        images = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        labels = list(labels_path.glob("*.txt"))

        self.logger.info(f"Found {len(images)} images and {len(labels)} labels.")

        self.samples = self._get_file_pairs(images, labels)
        self.samples = self._filter_for_valid_labels(self.samples)

        self.logger.info(f"{len(self.samples)} image and label pairs exist.")

    def _get_file_pairs(
        self, images: list[Path], labels: list[Path]
    ) -> list[tuple[Path, Path]]:
        """
        Keeps only image and label files that form a pair. A pair requires the image
        and label file to have the same stem.

        Parameters
        ----------
        images:
            A collection of paths to images.

        labels:
            A collection of paths to label files.

        Output
        ------
        samples:
            A list of samples. Each sample is structured as `(image_file, label_file)`.
        """
        stem_to_image_file = {im.stem: im for im in images}
        stem_to_label_file = {lab.stem: lab for lab in labels}

        samples: list[tuple[Path, Path]] = []

        for stem, image_file in stem_to_image_file.items():
            if stem not in stem_to_label_file:
                self.logger.warning(f"Missing label file for {image_file}.")
                continue
            samples.append((image_file, stem_to_label_file[stem]))

        return samples

    def _filter_for_valid_labels(
        self, samples: list[tuple[Path, Path]]
    ) -> list[tuple[Path, Path]]:
        """
        Filter out samples with invalid label files.

        Parameters
        ----------
        samples:
            Image and label file pairs.

        Returns
        -------
        valid_samples:
            Image and label file pairs where the label file contains no "errors".
        """

        valid_samples: list[tuple[Path, Path]] = []
        for image, label in samples:
            try:
                FrameLabels.from_file(
                    label, self.raw_class_names, self.dtype, self.device
                )
                valid_samples.append((image, label))
            except ValueError as err:
                self.logger.warning(f"Error in {label}...")
                self.logger.debug(err)

        return valid_samples

    def __len__(self) -> int:
        return len(self.samples)
