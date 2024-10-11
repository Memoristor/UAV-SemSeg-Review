#coding=utf-8

from torch import nn
from segmentation_models_pytorch import DeepLabV3Plus


__all__ = [
    'DeepLabV3PlusResNet50',
    'DeepLabV3PlusResNet101',
]


class DeepLabV3PlusResNet50(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}


class DeepLabV3PlusResNet101(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}
    