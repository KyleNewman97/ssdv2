from torch import nn

from ssdv2.models.backbones import ConvNeXtBackbone
from ssdv2.models.necks import ConvNeXtNeck


class SSDv2(nn.Module):
    def __init__(self):
        self.backbone = ConvNeXtBackbone(depths=[3, 3, 3, 3], dims=[48, 96, 192, 384])
        self.neck = ConvNeXtNeck(depths=[3, 3, 3], dims=[576, 672, 720])
