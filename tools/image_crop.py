# Author: Yahui Liu <yahui.liu@unitn.it>

"""
Usage:
  python3 image_crop.py --image_file <image_path> --save_path <save_folder>
"""

import cv2, os
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, default='../datasets/RoadNet/Ottawa-Dataset/1/Ottawa-1.tif', help='/path/to/image')
parser.add_argument('--save_path', type=str, default='../datasets/RoadNet/train_image')
parser.add_argument('--step', type=int, default=128, help='128 for training, 256 for testing')
args = parser.parse_args()

IMG_READ_MODE = cv2.IMREAD_UNCHANGED
PNG_SAVE_MODE = [cv2.IMWRITE_PNG_COMPRESSION, 0]
H, W = 512, 512
step = args.step

def crop_info(im_shape, sz=(H,W), step=step):
    new_h = im_shape[0] / step
    offset_h = im_shape[0] % step
    if offset_h > 0:
        new_h += 1
        offset_h = step - offset_h
    new_w = im_shape[1] / step
    offset_w = im_shape[1] % step
    if offset_w > 0:
        new_w += 1
        offset_w = step - offset_w
    return int(new_h), int(new_w), offset_h, offset_w

def imageCrop(im_file, save_path):
    assert os.path.isdir(save_path)
    # get the image index 
    fname = im_file.split('/')[-2]
    # load image and calculate cropping information
    im = cv2.imread(im_file, IMG_READ_MODE)
    s = im.shape
    new_h, new_w, offset_h, offset_w = crop_info(s)
    # save cropping information
    fp = open(os.path.join(save_path, '{}.info'.format(fname)), 'w')
    fp.write(str(new_h)+' '+str(new_w)+' '+str(offset_h)+' '+str(offset_w))
    fp.close()
    print("cropping info: ", new_h, new_w, offset_h, offset_w)
    # crop and save
    h, w = 0, 0
    for i in range(new_h):
        h = i * step
        if i == new_h-1:
            h -= offset_h
        for j in range(new_w):
            w = j * step
            if j == new_w-1:
                w -= offset_w
            im_roi = im[h:h+H, w:w+W, :]
            cv2.imwrite(os.path.join(save_path, 
                "{}-{}-{}.png".format(fname, i, j)), 
                im_roi, PNG_SAVE_MODE)

if __name__ == '__main__':
    print(args.image_file)
    imageCrop(args.image_file, args.save_path)
