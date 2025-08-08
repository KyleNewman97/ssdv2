from pathlib import Path

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    dataset_dir: Path = Field(description="Root dataset directory.")

    num_epochs: int = Field(default=100)
    batch_size: int = Field(default=32)
