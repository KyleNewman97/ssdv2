from pathlib import Path

from torch.utils.data import Dataset


class DatasetSampler(Dataset):
    """
    Responsible for sampling image and label pairs from a dataset.
    """

    def __init__(self, images_path: Path, labels_path: Path):
        pass
