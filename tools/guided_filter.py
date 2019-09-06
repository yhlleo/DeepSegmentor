# Author: Yahui Liu <yahui.liu@unitn.it>

import os
import glob
import cv2
from cv2.ximgproc import guidedFilter
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='deepcrack')
parser.add_argument('--results_dir', type=str, default='../results')
parser.add_argument('--suffix_fused', type=str, default='fused', help='Suffix of predicted file name')
parser.add_argument('--suffix_sideout', type=str, default='side1', help='Suffix of side output file name')
parser.add_argument('--suffix_output', type=str, default='gf', help='Suffix of refined results')
args = parser.parse_args()


if __name__ == '__main__':
    results_dir  = os.path.join(args.results_dir, args.model_name, 'test_latest', 'images')
    fused_list   = glob.glob(os.path.join(results_dir, '*{}.jpg'.format(args.suffix_fused)))
    sideout_list = [ll.replace(args.suffix_fused, args.suffix_sideout) for ll in fused_list]

    for ff, ss in zip(fused_list, sideout_list):
        img_fused = cv2.imread(ff, cv2.IMREAD_UNCHANGED)
        img_side  = cv2.imread(ss, cv2.IMREAD_UNCHANGED)

        img_out   = np.zeros(img_fused.shape, dtype='uint8')
        guidedFilter(img_fused, img_side, img_out, radius=5, eps=1e-6*255*255)

        cv2.write(ff.replace(args.suffix_fused, args.suffix_output), img_gf, 
            [cv2.IMWRITE_PNG_COMPRESSION, 0])