#coding=utf-8

import numpy as np
from torch import nn
import torch
import os

from models.vit_seg.vit_seg_modeling import VisionTransformer as ViT_seg
from models.vit_seg.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


__all__ = [
    'TransUNetR50ViTB16', 
    'TransUNetR50ViTL16',
    'TransUNetViTB16',
]


class TransUNet(nn.Module):
    
    def __init__(self, vit_name, num_class, img_size=224, n_skip=3, vit_patches_size=16):
        super(TransUNet, self).__init__()
        
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_classes = num_class
        config_vit.n_skip = n_skip
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
        self.net = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
        if os.path.exists(config_vit.pretrained_path):
            print(f'found weight at: {config_vit.pretrained_path}')
            self.net.load_from(weights=np.load(config_vit.pretrained_path))
        else:
            print(f'can not load weight from: {config_vit.pretrained_path}')
            
    def forward(self, x):
        return {'lbl': self.net(x)}
    

class TransUNetR50ViTB16(TransUNet):
    
    def __init__(self, num_class, img_size=512, n_skip=3, vit_patches_size=16):
        super(TransUNetR50ViTB16, self).__init__('R50-ViT-B_16', num_class, img_size, n_skip, vit_patches_size)
            

class TransUNetR50ViTL16(TransUNet):
    
    def __init__(self, num_class, img_size=512, n_skip=3, vit_patches_size=16):
        super(TransUNetR50ViTL16, self).__init__('R50-ViT-L_16', num_class, img_size, n_skip, vit_patches_size)
              

class TransUNetViTB16(TransUNet):
    
    def __init__(self, num_class, img_size=224, n_skip=3, vit_patches_size=16):
        super(TransUNetViTB16, self).__init__('ViT-B_16', num_class, img_size, n_skip, vit_patches_size)
