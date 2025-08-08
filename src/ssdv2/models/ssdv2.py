from torch import Tensor, nn

from ssdv2.models.backbones import ConvNeXtBackbone
from ssdv2.models.heads import FCOSHead
from ssdv2.models.necks import ConvNeXtNeck
from ssdv2.structs.configs import TrainConfig


class SSDv2(nn.Module):
    def __init__(self, num_classes: int):
        nn.Module.__init__(self)

        self.backbone = ConvNeXtBackbone(depths=[3, 3, 3, 3], dims=[64, 128, 256, 512])
        self.neck = ConvNeXtNeck(depths=[0, 3, 3, 3], fm_channels=[64, 128, 256, 512])
        self.head = FCOSHead(num_classes, self.neck.out_fm_channels)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        backbone_fms = self.backbone.forward(images)
        neck_fms = self.neck.forward(backbone_fms)
        return self.head.forward(neck_fms)

    def fit(self, config: TrainConfig):
        pass

    # def _create_data_loaders(
    #     self, config: TrainConfig
    # ) -> tuple[DataLoader, DataLoader]:
    #     """
    #     Creates the training and validation data loaders.
    #     """
    #     # Create the transform to use with all images being fed into the model
    #     transform = LetterboxTransform(
    #         config.image_width, config.image_height, config.dtype
    #     )

    #     # Create the collate function
    #     collate_func = partial(TrainUtils.batch_collate_func, device=self.device)

    #     # Create the training dataset loader
    #     train_dataset = SSDDataset(
    #         config.train_images_dir,
    #         config.train_labels_dir,
    #         config.num_classes,
    #         transform,
    #         None,
    #         self.device,
    #         config.dtype,
    #     )
    #     train_loader = DataLoader(
    #         train_dataset, config.batch_size, shuffle=True, collate_fn=collate_func
    #     )

    #     # Create the validation dataset loader
    #     val_dataset = SSDDataset(
    #         config.val_images_dir,
    #         config.val_labels_dir,
    #         config.num_classes,
    #         transform,
    #         None,
    #         self.device,
    #         config.dtype,
    #     )
    #     val_loader = DataLoader(
    #         val_dataset, config.batch_size, shuffle=False, collate_fn=collate_func
    #     )

    #     return train_loader, val_loader
