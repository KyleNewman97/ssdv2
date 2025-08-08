import torch
from pydantic import BaseModel
from torch import Tensor, nn
from torchvision.transforms.functional import resize

from ssdv2.structs import FrameLabels


class LetterboxTransform(nn.Module):
    """
    A transform that letterboxes an image. Specifically, this downscales the image to
    fit into the specified `width` and `height`, whilst maintaining the image's original
    aspect ratio. This image is then centered within a tensor of shape
    `(height, width)`. Any remaining pixels are zeroed.
    """

    def __init__(self, width: int, height: int):
        """
        Parameters
        ----------
        width:
            The width to make the output images (in pixels).

        height:
            The height to make the output images (in pixels).
        """
        nn.Module.__init__(self)

        self.desired_width = width
        self.desired_height = height

    def _calculate_transform_params(self, image_width: int, image_height: int):
        """
        Calculate parameters required to perform the letterbox operation.

        Parameters
        ----------
        image_width:
            The width of the input image in pixels.

        image_height:
            The height of the input image in pixels.

        Returns
        -------
        params:
            A collection of parameters that are used to perform the letterbox operation
            on the image and its bounding boxes.
        """
        desired_wh_ratio = self.desired_width / self.desired_height
        image_wh_ratio = image_width / image_height

        # Find the new width and height of the image after resizing it to fit into the
        # desired width and height - without loosing the aspect ratio
        if desired_wh_ratio <= image_wh_ratio:
            # Find the dimensions when we are bound by the width
            new_width = self.desired_width
            new_height = int(new_width / image_wh_ratio)
        else:
            # Find the dimensions when we are bound by the height
            new_height = self.desired_height
            new_width = int(new_height * image_wh_ratio)

        # Determine x and y start and ends
        x_start = (self.desired_width - new_width) // 2
        x_end = x_start + new_width
        y_start = (self.desired_height - new_height) // 2
        y_end = y_start + new_height

        return TransformParams(
            desired_wh_ratio=desired_wh_ratio,
            image_wh_ratio=image_wh_ratio,
            new_width=new_width,
            new_height=new_height,
            x_start=x_start,
            x_end=x_end,
            y_start=y_start,
            y_end=y_end,
        )

    def transform_image(self, image: Tensor) -> Tensor:
        """
        Applies the letterbox transformation to the input image.

        Parameters
        ----------
        image:
            Image tensor to apply the letterbox transform to. The shape should be:
                `(num_channels, height, width)`

        Returns
        -------
        letterboxed_image:
            A letterboxed version of the original image.
        """
        dtype = image.dtype
        device = image.device

        params = self._calculate_transform_params(image.shape[2], image.shape[1])

        # Create the output image
        resized_image = resize(image, [params.new_height, params.new_width])
        desired_shape = (image.shape[0], self.desired_height, self.desired_width)
        output_image = torch.zeros(desired_shape, dtype=dtype, device=device)
        output_image[
            :, params.y_start : params.y_end, params.x_start : params.x_end
        ] = resized_image

        return output_image

    def transform_objects(
        self, objects: FrameLabels, image_width: int, image_height: int
    ) -> FrameLabels:
        """
        Applies the letterbox transform to the input objects.

        Parameters
        ----------
        objects:
            Objects to apply the letterbox transform to.

        image_width:
            The original width of the image in pixels (width before letterboxing).

        image_height:
            The original height of the image in pixels (height before letterboxing).

        Returns
        -------
        letterboxed_objects:
            Letterboxed versions of the original objects.
        """
        params = self._calculate_transform_params(image_width, image_height)

        # Adjust the object positions
        out_boxes = objects.boxes.clone()
        if params.desired_wh_ratio <= params.image_wh_ratio:
            # When bound by the width adjust the y values
            out_boxes[:, 1::2] *= params.new_height / self.desired_height
            out_boxes[:, 1] += params.y_start / self.desired_height
        else:
            # When bound by the height adjust the x values
            out_boxes[:, 0::2] *= params.new_width / self.desired_width
            out_boxes[:, 0] += params.x_start / self.desired_width

        return FrameLabels(
            boxes=out_boxes,
            raw_class_ids=objects.raw_class_ids,
            raw_class_names=objects.raw_class_names,
        )

    def forward(
        self, image: Tensor, objects: FrameLabels
    ) -> tuple[Tensor, FrameLabels]:
        """
        Applies the letterbox transform to both the image and the objects.

        Parameters
        ----------
        image:
            A single image tensor with dimensions of `(channels, height, width)`.

        objects:
            Labelled objects for the image.

        Returns
        -------
        output_image:
            A letterboxed version of the original input image. This will have a shape of
            `(channels, desired_height, desired_width)`.

        out_objects:
            A letterboxed version of the image's objects.
        """

        letterbox_image = self.transform_image(image)
        letterbox_objs = self.transform_objects(objects, image.shape[2], image.shape[1])

        return letterbox_image, letterbox_objs


class TransformParams(BaseModel):
    desired_wh_ratio: float
    image_wh_ratio: float
    new_width: int
    new_height: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int
