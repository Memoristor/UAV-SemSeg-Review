# coding=utf-8

import os
import sys
from typing import Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets.basic import BasicDataset
import numpy as np


__all__ = ['UAVid']


class UAVid(BasicDataset):
    """
    UAVid dataset

    Params:
        root_path: str. The root path to the data folder, which contains `seq(x)/`
        image_size: tuple. The size of output image, which format is (H, W)
        phase: str. Indicates that the dataset is used for {`train`, `test`, `valid`}
        mean: list. The mean value of normed RGB, obtained by statistic
        std: list. The std value of normed RGB, obtained by statistic
        div_std: bool. (default True). Whether the normed data will be divided by `std`
        class_rgb: dict. The classes' RGB value, e.g {'IS': [255, 255, 255], 'BD': [0, 0, 255]}
        weight: list. The weight of each class, obtained by statistic
        chw_format: bool (default True). If True, the output data's format is CxHxW, otherwise HxWxC
    """
    def __init__(
        self, 
        root_path: str, 
        image_size: Union[int, list, tuple], 
        phase: str, 
        mean=(0.4183057, 0.43552268, 0.3968076), 
        std=(0.25573143, 0.2467472, 0.25252375), 
        div_std=True,
        class_rgb = {
            'Background clutter': (0, 0, 0),
            'Building': (128, 0, 0),
            'Road': (128, 64, 128),
            'Tree': (0, 128, 0),
            'Low vegetation': (128, 128, 0),
            'Moving car': (64, 0, 128),
            'Static car': (192, 0, 192),
            'Human': (64, 64, 0),
        },
        weight=(1.83064886, 1.72729677, 1.86549853, 1.74068761, 1.86066169, 1.99004619, 1.98777819, 1.99738215),
        chw_format=True, 
    ):
        super(UAVid, self).__init__(
            root_path=root_path,
            image_size=image_size,
            phase=phase,
            mean=mean,
            std=std,
            div_std=div_std,
            class_rgb=class_rgb,
            weight=weight,
            chw_format=chw_format,
        )
        
        # Get colors of each class
        self.num_class = len(self.class_rgb)

        # Get path of all images, labels
        self.img_path = []
        self.lbl_path = []
        self.img_name = []
        for item in os.listdir(os.path.join(root_path, 'Images')):
            self.img_path.append(os.path.join(root_path, 'Images', item))
            self.lbl_path.append(os.path.join(root_path, 'Labels', item))
            self.img_name.append(item)
