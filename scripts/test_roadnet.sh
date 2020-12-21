GPU_IDS=$1

DATAROOT=./datasets/RoadNet
NAME=roadnet
MODEL=roadnet
DATASET_MODE=roadnet

BATCH_SIZE=1
LOAD_WIDTH=512
LOAD_HEIGHT=512

NUM_TEST=100

python3 test.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --batch_size ${BATCH_SIZE} \
  --output_nc 1 \
  --norm batch \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --use_selu 0 \
  --num_test ${NUM_TEST}
