from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor


class FrameLabels(BaseModel):
    """
    Object labels for an individual frame.
    """

    boxes: Tensor = Field(
        description=(
            "Object bounding boxes, with shape: `(num_objects, 4)`. The last dimension"
            "contains `(cx, cy, w, h)`. All values are normalised between 0 and 1."
        )
    )
    raw_class_ids: Tensor = Field(
        description=(
            "Object class IDs, with shape: `(num_objects,)`. These class IDs"
            "appear like those in the label file."
        )
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_file(
        cls, file: Path, dtype: torch.dtype, device: torch.device
    ) -> "FrameLabels":
        """
        Initialise a `FrameLabels` object from a label file.

        Parameters
        ----------
        file:
            Path to a file containing object labels. The contents of this file should be
            structured as:
                ```
                class_id, cx, cy, w, h
                ...
                ```

        dtype:
            The type to use within the boxes tensor.

        device:
            The device to put the boxes and class IDs onto.

        Returns
        -------
        objects:
            A `FrameLabels` object containing the objects in the label file.
        """
        with open(file, "r") as fp:
            lines = fp.read().strip().split("\n")

            # Extract the class ID and box from each line
            class_ids: list[int] = []
            boxes: list[tuple[float, float, float, float]] = []
            for line in lines:
                elements = line.strip().split(" ")

                # Only allow valid label rows
                if len(elements) != 5:
                    raise ValueError(f"{file}")

                class_id = int(elements[0])
                center_x = float(elements[1])
                center_y = float(elements[2])
                width = float(elements[3])
                height = float(elements[4])
                class_ids.append(class_id)
                boxes.append((center_x, center_y, width, height))

        return FrameLabels(
            boxes=torch.tensor(boxes, dtype=dtype, device=device),
            raw_class_ids=torch.tensor(class_ids, dtype=torch.int, device=device),
        )

    def to_file(self, file: Path):
        """
        Writes the bounding boxes to a label file.

        Parameters
        ----------
        file:
            The file to write the objects to. The file's contents is structured as:
                ```
                class_id, cx, cy, w, h
                ...
                ```
        """
        with open(file, "w") as fp:
            for class_id, box in zip(self.raw_class_ids, self.boxes, strict=True):
                # Place boxes on the CPU
                class_id_int = int(class_id.cpu().item())
                box_list = box.cpu().tolist()

                fp.write(
                    f"{class_id_int} {box_list[0]} {box_list[1]} {box_list[2]} "
                    f"{box_list[3]}\n"
                )

    def contains_class(self, raw_class_id: int) -> bool:
        """
        Returns a boolean indicating whether this frame/image contains an objects of the
        specified class.

        Parameters
        ----------
        raw_class_id:
            The raw class ID associated with the class we want to see if the frame
            contains.

        Returns
        -------
        contains:
            Whether the frame contains an object of the specified type.
        """
        return raw_class_id in self.raw_class_ids
