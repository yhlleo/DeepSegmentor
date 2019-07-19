GPU_IDS=$1

DATAROOT=./datasets/RoadNet
NAME=roadnet
MODEL=roadnet
DATASET_MODE=roadnet

BATCH_SIZE=1
LOAD_WIDTH=512
LOAD_HEIGHT=512

USE_SELU=0
NITER=200
NITER_DECAY=200

python3 train.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --batch_size ${BATCH_SIZE} \
  --output_nc 1 \
  --use_selu ${USE_SELU} \
  --lr_decay_iters 40 \
  --lr_policy step \
  --load_width ${LOAD_WIDTH} \
  --load_height ${LOAD_HEIGHT} \
  --no_flip 0 \
  --norm batch \
  --display_id 0
