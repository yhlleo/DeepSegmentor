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
parser.add_argument('--thresh', type=float, default=0.31, help='using the best threshold')
parser.add_argument('--epsilon', type=float, default=0.065, help='eps = 1e-6*255*255')
parser.add_argument('--radius', type=int, default=5)
parser.add_argument('--suffix_fused', type=str, default='fused', help='Suffix of predicted file name')
parser.add_argument('--suffix_sideout', type=str, default='side1', help='Suffix of side output file name')
parser.add_argument('--suffix_output', type=str, default='gf', help='Suffix of refined results')
args = parser.parse_args()


if __name__ == '__main__':
    results_dir  = os.path.join(args.results_dir, args.model_name, 'test_latest', 'images')
    fused_list   = glob.glob(os.path.join(results_dir, '*{}.png'.format(args.suffix_fused)))
    sideout_list = [ll.replace(args.suffix_fused, args.suffix_sideout) for ll in fused_list]

    for ff, ss in zip(fused_list, sideout_list):
        img_fused = (cv2.imread(ff, cv2.IMREAD_UNCHANGED)>args.thresh*255).astype('uint8')*255
        img_side  = cv2.imread(ss, cv2.IMREAD_UNCHANGED)

        img_out = guidedFilter(img_side, img_fused, radius=args.radius, eps=args.epsilon)

        cv2.imwrite(ff.replace(args.suffix_fused, args.suffix_output), img_out, 
            [cv2.IMWRITE_PNG_COMPRESSION, 0])