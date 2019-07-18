#!/bin/bash
for i in `seq 2 15`
do
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/Ottawa-$i.tif --save_path ../datasets/RoadNet/train_image
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/segmentation.png --save_path ../datasets/RoadNet/train_segment
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/edge.png --save_path ../datasets/RoadNet/train_edge
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/centerline.png --save_path ../datasets/RoadNet/train_centerline
done