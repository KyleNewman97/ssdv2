from torch import Tensor, nn

from ssdv2.models.backbones import ConvNeXtBackbone
from ssdv2.models.heads import FCOSHead
from ssdv2.models.necks import ConvNeXtNeck


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
