# Author: Yahui Liu <yahui.liu@unitn.it>

import numpy as np
import torch

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)>127).long()