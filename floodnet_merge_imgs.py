# coding=utf-8

from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import math
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Merge Images')
    parser.add_argument('--model', type=str, nargs='+', help='The model will be used')
    parser.add_argument('--crop_height', type=int, default=1024, help='Cropped image height')
    parser.add_argument('--crop_width', type=int, default=1024, help='Cropped image width')
    parser.add_argument('--vert_stride', type=int, default=1024, help='Vertical crop stride')
    parser.add_argument('--hori_stride', type=int, default=1024, help='Horizontal crop stride')
    parser.add_argument('--image_path', type=str, help='The images path', default='./data/FloodNet-Supervised_v1/test')
    parser.add_argument('--output_path', type=str, help='The output path', default='./output')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # root path
    img_root = args.image_path
    out_root = args.output_path

    # creat directories
    for model in args.model:
        prd_root = os.path.join(out_root, "FloodNet", model, 'lr_0.0001_wd_0.0001_bs_8_sd_369', "images", "lbl")
        mrg_root = os.path.join(out_root, "FloodNet", model, 'lr_0.0001_wd_0.0001_bs_8_sd_369', 'merge')

        # get image and label path
        img_params = []
        for item in os.listdir(os.path.join(img_root, 'org-img')):
            img_params.append((os.path.join(img_root, 'org-img', item), item.split('.')[0]))
        
        # crop images
        for i in tqdm(range(len(img_params))):
            fpath, fname = img_params[i]
            img = Image.open(fpath)
            w, h = img.size
            
            nw = math.ceil((w - args.crop_width) / args.hori_stride)
            nh = math.ceil((h - args.crop_height) / args.vert_stride)
            
            bw = nw * args.hori_stride + args.crop_width
            bh = nh * args.vert_stride + args.crop_height
            img_whiteboard = np.ones((bh, bw, 3), dtype=np.uint8) * 0
            
            top = (bh - h) // 2  
            left = (bw - w) // 2
            # img_whiteboard[top:top+h, left:left+w, ...] = np.array(img, dtype=np.uint8)

            img = Image.fromarray(img_whiteboard)

            loc_w = [lw * args.hori_stride for lw in range(0, (bw - args.crop_width) // args.hori_stride + 1)]
            loc_h = [lh * args.vert_stride for lh in range(0, (bh - args.crop_height) // args.vert_stride + 1)]

            crop_loc = []
            for lw in loc_w:
                for lh in loc_h:
                    crop_loc.append((lw, lh, lw + args.crop_width, lh + args.crop_height))

            for i, loc in enumerate(crop_loc):
                prd_img = Image.open(os.path.join(prd_root, f'{fname}_{i+1}.jpg'))
                prd_img = prd_img.resize((args.crop_width, args.crop_height))
                
                img_whiteboard[loc[1]:loc[3], loc[0]:loc[2], ...] = np.array(prd_img, dtype=np.uint8)
                
            img_whiteboard = img_whiteboard[top:top+h, left:left+w, ...]
            img_whiteboard = Image.fromarray(img_whiteboard)
            
            os.makedirs(mrg_root, exist_ok=True)
            
            img_whiteboard.save(os.path.join(mrg_root, f'{fname}.jpg'))
            


