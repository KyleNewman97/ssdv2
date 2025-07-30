from pathlib import Path

import fiftyone as fo
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torchvision.ops import box_convert


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
    raw_class_names: dict[int, str] = Field(
        description="A map from class ID to class name."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def object_class_names(self) -> list[str]:
        """
        List of the class names of objects in the frame.
        """
        class_ids = self.raw_class_ids.cpu().tolist()
        return [self.raw_class_names[id] for id in class_ids]

    @property
    def fo_detections(self) -> fo.Detections:
        """
        The objects as fiftyone detections.
        """
        class_names = self.object_class_names
        boxes = box_convert(self.boxes, "cxcywh", "xywh").cpu().tolist()

        # Create the fiftyone detections object
        return fo.Detections(
            detections=[
                fo.Detection(label=class_name, bounding_box=box)
                for class_name, box in zip(class_names, boxes, strict=True)
            ]
        )

    @classmethod
    def from_file(
        cls,
        file: Path,
        raw_class_names: dict[int, str],
        dtype: torch.dtype,
        device: torch.device,
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

        raw_class_names:
            A map of raw class IDs to class names.

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

                # Check the class ID is valid
                if class_id not in raw_class_names:
                    raise ValueError(f"Invalid class ID: {class_id}")

                class_ids.append(class_id)
                boxes.append((center_x, center_y, width, height))

        boxes_tensor = torch.tensor(boxes, dtype=dtype, device=device)

        # Check the boxes are within the range of [0, 1]
        if boxes_tensor.min() < 0 or 1 < boxes_tensor.max():
            raise ValueError("Box coords must be between 0 and 1 inclusive.")

        return FrameLabels(
            boxes=boxes_tensor,
            raw_class_ids=torch.tensor(class_ids, dtype=torch.int, device=device),
            raw_class_names=raw_class_names,
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

    def change_classes(self, raw_class_ids: Tensor) -> "FrameLabels":
        """
        Removes objects that aren't in the specified classes and changes the class
        names to only contain these classes. This process results in the class IDs being
        changed so that they remain sequential.

        Parameters
        ----------
        raw_class_ids:
            The classes we wish to keep.
        """
        # Filter out all objects that aren't of the specified classes
        mask = torch.isin(self.raw_class_ids, raw_class_ids)
        boxes = self.boxes[mask, :]
        masked_classes = self.raw_class_ids[mask]

        # Change the class IDs
        raw_class_ids_list = raw_class_ids.cpu().tolist()
        old_to_new_class_ids = {id: idx for idx, id in enumerate(raw_class_ids_list)}
        new_classes = masked_classes.clone()
        for old, new in old_to_new_class_ids.items():
            new_classes[masked_classes == old] = new

        # Create the new class names dictionary
        class_names = {
            idx: self.raw_class_names[id] for idx, id in enumerate(raw_class_ids_list)
        }

        return FrameLabels(
            boxes=boxes, raw_class_ids=new_classes, raw_class_names=class_names
        )

    def __len__(self) -> int:
        return self.boxes.shape[0]
