#coding=utf-8

from torch import nn
from segmentation_models_pytorch import Unet
from segmentation_models_pytorch import UnetPlusPlus


__all__ = [
    'UNetResNet50',
    'UNetResNet101',
    'UNetPlusPlusResNet50',
    'UNetPlusPlusResNet101',
]


class UNetResNet50(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}


class UNetResNet101(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = Unet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}
    

class UNetPlusPlusResNet50(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}


class UNetPlusPlusResNet101(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.pretrained_net = UnetPlusPlus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            classes=num_class,
        )

    def forward(self, x):
        output = self.pretrained_net(x)
        return {'lbl': output}


if __name__ == '__main__':
    from torchstat import stat

    model = UNetResNet50(num_class=6)
    stat(model, (3, 512, 512))
