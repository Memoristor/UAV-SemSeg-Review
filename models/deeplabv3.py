#coding=utf-8

from torch import nn
from segmentation_models_pytorch import DeepLabV3


__all__ = [
    'DeepLabV3ResNet50',
    'DeepLabV3ResNet101',
]


class DeepLabV3ResNet50(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = DeepLabV3(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}


class DeepLabV3ResNet101(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = DeepLabV3(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}
    