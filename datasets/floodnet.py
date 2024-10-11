# coding=utf-8

import os
import sys
from typing import Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets.basic import BasicDataset
import numpy as np


__all__ = ['FloodNet']


class FloodNet(BasicDataset):
    """
    FloodNet dataset

    Params:
        root_path: str. The root path to the data folder, which contains `Images/` and `Labels/`
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
        mean=(0.40885398, 0.44664785, 0.33997318), 
        std=(0.20739624, 0.19265558, 0.20844948), 
        div_std=True,
        class_rgb = {
            'Background': (0, 0, 0),
            'Building-flooded': (255, 71, 0), 
            'Building-non-flooded': (180, 120, 120), 
            'Road-flooded': (160, 150, 20), 
            'Road-non-flooded': (140, 140, 140), 
            'Water': (61, 230, 250), 
            'Tree': (0, 82, 255), 
            'Vehicle': (255, 0, 245), 
            'Pool': (255, 0, 0), 
            'Grass': (4, 250, 7),
        },
        weight=(1.98294835, 1.98469099, 1.96779831, 1.97237648, 1.9450856, 1.89001808, 1.8264362, 1.99586401, 1.99564512, 1.43913686),
        chw_format=True, 
    ):
        super(FloodNet, self).__init__(
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
        for item in os.listdir(os.path.join(root_path, 'org-img')):
            self.img_path.append(os.path.join(root_path, 'org-img', item))
            self.lbl_path.append(os.path.join(root_path, 'label-img', item.split('.')[0] + '.png')) # crop image set
            # self.lbl_path.append(os.path.join(root_path, 'label-img', item.split('.')[0] + '_lab.png'))
            self.img_name.append(item)


    def encode_lbl(self, label):
        """
        The image is encoded from RGB format to the format required by the training model

        Params:
            label: 2-D numpy array. RGB images to be encoded. Note that the ignored
            category is encoded as 255

        Return:
            return the encoded label
        """
        return label

    def decode_lbl(self, label):
        """
        The image is decoded from the format required by the training model into RGB format

        Params:
            label: 2-D numpy array. Label to de decoded. Note that the ignored
            category will be decoded as 255

        Return:
            return the decoded label
        """
        lbl = np.ones((*label.shape[0:2], 3), dtype=np.uint8) * 255
        
        for i, cls in enumerate(self.class_rgb.keys()):
            lbl[label == i, :] = self.class_rgb[cls]

        if self.chw_format:
            lbl = lbl.transpose([2, 0, 1])

        return lbl
    
