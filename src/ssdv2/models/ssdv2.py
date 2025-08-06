from torch import nn

from ssdv2.models.backbones import ConvNeXtBackbone
from ssdv2.models.heads import FCOSHead
from ssdv2.models.necks import ConvNeXtNeck


class SSDv2(nn.Module):
    def __init__(self, num_classes: int):
        self.backbone = ConvNeXtBackbone(depths=[3, 3, 3, 3], dims=[48, 96, 192, 384])
        self.neck = ConvNeXtNeck(depths=[0, 3, 3, 3], fm_channels=[384, 192, 96, 48])
        self.head = FCOSHead(num_classes, self.neck.out_fm_channels)
