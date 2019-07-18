# Author: Yahui Liu <yahui.liu@unitn.it>

import torch
import cv2
import numpy as np
import random

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()

def get_params(angle_scope=45, scale_scope=0.2, shift_scope=32):
    angle = random.uniform(-angle_scope, angle_scope)
    scale = random.uniform(1-scale_scope, 1+scale_scope)    
    shift_x, shift_y = random.randint(-shift_scope, shift_scope), random.randint(-shift_scope, shift_scope)
    return angle, scale, (shift_x, shift_y)

def affine_transform(img, angle, scale, shift, w, h):
    # Rotation
    MR = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    img = cv2.warpAffine(img, MR, (w, h))
    # Translation
    MT = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
    img = cv2.warpAffine(img, MT, (w, h))
    return img

def convert_from_color_annotation(arr_3d):
    arr_3d = arr_3d.astype('uint8')
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = {(255,255,255): 0, (0,0,255): 255}
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d