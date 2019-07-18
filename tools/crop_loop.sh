#!/bin/bash
mkdir ../datasets/RoadNet

# train
mkdir ../datasets/RoadNet/train_image
mkdir ../datasets/RoadNet/train_segment
mkdir ../datasets/RoadNet/train_edge
mkdir ../datasets/RoadNet/train_centerline
for i in `seq 2 15`
do
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/Ottawa-$i.tif --save_path ../datasets/RoadNet/train_image
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/segmentation.png --save_path ../datasets/RoadNet/train_segment
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/edge.png --save_path ../datasets/RoadNet/train_edge
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/centerline.png --save_path ../datasets/RoadNet/train_centerline
done

# test
mkdir ../datasets/RoadNet/test_image
mkdir ../datasets/RoadNet/test_segment
mkdir ../datasets/RoadNet/test_edge
mkdir ../datasets/RoadNet/test_centerline
for i in 1 16 17 18 19 20
do
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/Ottawa-$i.tif --save_path ../datasets/RoadNet/test_image
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/segmentation.png --save_path ../datasets/RoadNet/test_segment
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/edge.png --save_path ../datasets/RoadNet/test_edge
    python3 image_crop.py --image_file ../datasets/Ottawa-Dataset/$i/centerline.png --save_path ../datasets/RoadNet/test_centerline
done
