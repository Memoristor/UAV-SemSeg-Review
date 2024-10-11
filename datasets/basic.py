# coding=utf-8

from torch.utils.data import Dataset
from datasets import transformers
from PIL import Image
import numpy as np
from typing import Union


__all__ = ['BasicDataset']


def str_to_arr(string, max_len=128):
    """
    Convert string to numpy array (ASCII)
    """
    arr = np.zeros(max_len, dtype=np.uint8)
    for i, s in enumerate(string):
        if i < max_len:
            arr[i] = ord(s)
        else:
            break
    return arr


class BasicDataset(Dataset):
    """
    Basic dataset

    Params:
        root_path: str. The root path to the data folder, which contains `image/` and `label/`
        image_size: tuple. The size of output image, which format is (H, W)
        phase: str. Indicates that the dataset is used for {`train`, `test`, `valid`}
        mean: list (default (0, 0, 0)). The mean value of normed RGB
        std: list (default (1, 1, 1)). The std value of normed RGB
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
        mean=(0, 0, 0), 
        std=(1, 1, 1), 
        div_std=True,
        class_rgb={},
        weight=None,
        chw_format=True, 
    ):
        self.root_path = root_path
        
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
            
        self.phase = phase
        self.mean = mean
        self.std = std
        self.div_std = div_std
        self.class_rgb = class_rgb
        self.weight = weight
        self.chw_format = chw_format
        
        # init colors of each class
        self.num_class = len(self.class_rgb)

        # init all images, labels, names
        self.img_path = []
        self.lbl_path = []
        self.img_name = []

        # init augmentation sequence
        if phase == 'train':
            self.aug_seq = transformers.Compose([
                transformers.RandomHorizontalFlip(),
                transformers.RandomVerticalFlip(),
                transformers.RandomScaleCropEx((self.image_size[1], self.image_size[0])),
                transformers.RandomGaussianBlur(),
                transformers.Normalize(self.mean, self.std, self.div_std),
                transformers.ToNumpy(),
            ])
        elif phase == "valid": 
            self.aug_seq = transformers.Compose([
                transformers.Resize((self.image_size[1], self.image_size[0])),
                transformers.Normalize(self.mean, self.std, self.div_std),
                transformers.ToNumpy(),
            ])
        elif phase == "test":
            self.aug_seq = transformers.Compose([
                transformers.Resize((self.image_size[1], self.image_size[0])),
                transformers.Normalize(self.mean, self.std, self.div_std),
                transformers.ToNumpy(),
            ])
        else:
            self.aug_seq = None

    def encode_lbl(self, label):
        """
        The image is encoded from RGB format to the format required by the training model

        Params:
            label: 3-D numpy array. RGB images to be encoded. Note that the ignored
            category is encoded as 255

        Return:
            return the encoded label
        """
        lbl = np.ones(label.shape[0:2], dtype=np.int16) * 255
        for i, cls in enumerate(self.class_rgb.keys()):
            lbl[np.where(np.all(label == self.class_rgb[cls], axis=-1))[:2]] = i
        return lbl

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
    
    def load_img(self, path):
        """Load image
        """
        return Image.open(path)
    
    def load_lbl(self, path):
        """Load label
        """
        return Image.open(path)

    def __getitem__(self, index):
        """
        Get item by index
        """
        # Open image and label
        img = self.load_img(self.img_path[index])
        lbl = self.load_lbl(self.lbl_path[index])
        
        assert img.size[0:2] == lbl.size[0:2]
        hw = img.size[0:2]

        # Data argumentation
        img, lbl = self.aug_seq(img, lbl)

        # Encode label
        lbl = self.encode_lbl(lbl)

        # Channel transpose
        if self.chw_format:
            img = img.transpose([2, 0, 1])

        return {
            'img': np.array(img, dtype=np.float32),
            'lbl': np.array(lbl, dtype=np.float32),
            'hw': np.array(hw, dtype=np.int32),
            'fnm': str_to_arr(self.img_name[index]),
        }

    def __len__(self):
        return len(self.img_path)

