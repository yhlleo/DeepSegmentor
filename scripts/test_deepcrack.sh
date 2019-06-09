GPU_IDS=$1

DATAROOT=./datasets/DeepCrack
NAME=deepcrack
MODEL=deepcrack
DATASET_MODE=deepcrack

BATCH_SIZE=1
NORM=batch

NUM_CLASSES=2

python3.5 test.py \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --dataset_mode ${DATASET_MODE} \
  --gpu_ids ${GPU_IDS} \
  --batch_size ${BATCH_SIZE} \
  --num_classes ${NUM_CLASSES} \
  --norm ${NORM} \
  --use_augment 0
