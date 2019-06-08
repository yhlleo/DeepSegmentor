# Author: Yahui Liu <yahui.liu@unitn.it>

import torch
import cv2
import numpy as np
import random

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy((np.array(img, dtype=np.uint8)/127).astype('int32')).long()


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