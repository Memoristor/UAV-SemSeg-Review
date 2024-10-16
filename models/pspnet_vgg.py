# coding=utf-8

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import *

__all__ = ['PSPNetVGG16', 'PSPNetVGG16BN', 'PSPNetVGG19', 'PSPNetVGG19BN']


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels=1024, pool_factors=(1, 2, 3, 6), batch_norm=True):
        super().__init__()
        self.spatial_blocks = []
        for pf in pool_factors:
            self.spatial_blocks += [self._make_spatial_block(in_channels, pf, batch_norm)]
        self.spatial_blocks = nn.ModuleList(self.spatial_blocks)

        bottleneck = []
        bottleneck += [nn.Conv2d(in_channels * (len(pool_factors) + 1), out_channels, kernel_size=1)]
        if batch_norm:
            bottleneck += [nn.BatchNorm2d(out_channels)]
        bottleneck += [nn.ReLU(inplace=True)]
        self.bottleneck = nn.Sequential(*bottleneck)

    def _make_spatial_block(self, in_channels, pool_factor, batch_norm):
        spatial_block = []
        spatial_block += [nn.AdaptiveAvgPool2d(output_size=(pool_factor, pool_factor))]
        spatial_block += [nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)]
        if batch_norm:
            spatial_block += [nn.BatchNorm2d(in_channels)]
        spatial_block += [nn.ReLU(inplace=True)]

        return nn.Sequential(*spatial_block)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pool_outs = [x]
        for block in self.spatial_blocks:
            pooled = block(x)
            pool_outs += [F.upsample(pooled, size=(h, w), mode='bilinear')]
        o = torch.cat(pool_outs, dim=1)
        o = self.bottleneck(o)
        return o

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PSPUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        layers = []
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.ReLU(inplace=True)]
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(x, size=(h, w), mode='bilinear')
        return self.layer(p)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PSPNet(torch.nn.Module):

    def __init__(self, num_class, pretrained_model, batch_norm=True, psp_out_feature=1024):
        super(PSPNet, self).__init__()
        self.features = pretrained_model.features

        # find out_channels of the top layer and define classifier
        for idx, m in reversed(list(enumerate(self.features.modules()))):
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels
                break

        self.PSP = PSPModule(channels, out_channels=psp_out_feature, batch_norm=batch_norm)
        h_psp_out_feature = int(psp_out_feature / 2)
        q_psp_out_feature = int(psp_out_feature / 4)
        e_psp_out_feature = int(psp_out_feature / 8)
        self.upsampling1 = PSPUpsampling(psp_out_feature, h_psp_out_feature, batch_norm=batch_norm)
        self.upsampling2 = PSPUpsampling(h_psp_out_feature, q_psp_out_feature, batch_norm=batch_norm)
        self.upsampling3 = PSPUpsampling(q_psp_out_feature, e_psp_out_feature, batch_norm=batch_norm)

        self.classifier = nn.Sequential(nn.Conv2d(e_psp_out_feature, num_class, kernel_size=1))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        for f in self.features:
            o = f(o)

        o = self.PSP(o)
        o = self.upsampling1(o)
        o = self.upsampling2(o)
        o = self.upsampling3(o)

        o = F.upsample(o, size=(x.shape[2], x.shape[3]), mode='bilinear')
        o = self.classifier(o)

        # return o
        return {'lbl': o}


class PSPNetVGG16(PSPNet):
    def __init__(self, num_class):
        super(PSPNetVGG16, self).__init__(num_class, pretrained_model=vgg16(pretrained=True), batch_norm=True)


class PSPNetVGG16BN(PSPNet):
    def __init__(self, num_class):
        super(PSPNetVGG16BN, self).__init__(num_class, pretrained_model=vgg16_bn(pretrained=True), batch_norm=True)


class PSPNetVGG19(PSPNet):
    def __init__(self, num_class):
        super(PSPNetVGG19, self).__init__(num_class, pretrained_model=vgg19(pretrained=True), batch_norm=True)


class PSPNetVGG19BN(PSPNet):
    def __init__(self, num_class):
        super(PSPNetVGG19BN, self).__init__(num_class, pretrained_model=vgg19_bn(pretrained=True), batch_norm=True)


if __name__ == '__main__':
    from torchstat import stat

    model = PSPNetVGG16(num_class=6)
    stat(model, (3, 512, 512))
