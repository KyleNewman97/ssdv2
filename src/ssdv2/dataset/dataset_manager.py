from pathlib import Path


class DatasetManager:
    def __init__(self, dataset_dir: Path):
        self._dataset_dir = dataset_dir

    @staticmethod
    def _verify_dataset_dir(dataset_dir: Path):
        """
        Verify that the directory structure is correct.
        """
