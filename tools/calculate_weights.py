# -*- using: utf-8 -*-
# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import glob
import cv2
import numpy as np
import statistics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='', 
    help='/path/to/segmentation')
args = parser.parse_args()

def get_weights(labels_dict):
    total_pixels = 0
    for lab in labels_dict:
        total_pixels += labels_dict[lab]
    for lab in labels_dict:
        labels_dict[lab] /= float(total_pixels)
    return labels_dict

def calculate_weights(im_path):
    assert os.path.isdir(im_path)
    img_list = glob.glob(os.path.join(im_path, '*.png'))
    labels_dict = {}
    for im_path in img_list:
        im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)
        labels, counts = np.unique(im, return_counts=True)
        for lab, cnt in zip(labels, counts):
            if lab not in labels_dict:
                labels_dict[lab] = 0
            labels_dict[lab] += cnt
    return get_weights(labels_dict)

def reverse_weight(w):
    """
    Median Frequency Balancing: alpha_c = median_freq/freq(c).
    median_freq is the median of these frequencies 
    freq(c) is the number of pixles of class c divided by the total number of pixels in images where c is present
    """
    assert len(w) > 0, "Expected a non-empty weight dict."
    values = [w[k] for k in w]
    if len(w) == 1:
        value = 1.0
    elif len(w) == 2:
        value = min(values)
    else:
        # Median Frequency Balancing
        value = statistics.median(values)
    for k in w:
        w[k] = value/(w[k]+1e-10)
    return w

if __name__ == '__main__':
    weights = calculate_weights(args.data_path)
    print(weights)
    # {0: 0.9708725873161764, 255: 0.02912741268382353}
    print(reverse_weight(weights))
    # {0: 0.030001272114749396, 255: 0.9999999965668079}