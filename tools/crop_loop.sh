#!/bin/bash
dataroot=../datasets/Ottawa-Dataset
save_folder=../datasets/RoadNet

# train
mkdir ${save_folder}/train_image
mkdir ${save_folder}/train_segment
mkdir ${save_folder}/train_edge
mkdir ${save_folder}/train_centerline
for i in `seq 2 15`
do
    python3 image_crop.py --image_file ${dataroot}/$i/Ottawa-$i.tif --save_path ${save_folder}/train_image
    python3 image_crop.py --image_file ${dataroot}/$i/segmentation.png --save_path ${save_folder}/train_segment
    python3 image_crop.py --image_file ${dataroot}/$i/edge.png --save_path ${save_folder}/train_edge
    python3 image_crop.py --image_file ${dataroot}/$i/centerline.png --save_path ${save_folder}/train_centerline
done

# test
mkdir ${save_folder}/test_image
mkdir ${save_folder}/test_segment
mkdir ${save_folder}/test_edge
mkdir ${save_folder}/test_centerline
for i in 1 16 17 18 19 20
do
    python3 image_crop.py --image_file ${dataroot}/$i/Ottawa-$i.tif --save_path ${save_folder}/test_image --step 256
    python3 image_crop.py --image_file ${dataroot}/$i/segmentation.png --save_path ${save_folder}/test_segment --step 256
    python3 image_crop.py --image_file ${dataroot}/$i/edge.png --save_path ${save_folder}/test_edge --step 256
    python3 image_crop.py --image_file ${dataroot}/$i/centerline.png --save_path ${save_folder}/test_centerline --step 256
done
