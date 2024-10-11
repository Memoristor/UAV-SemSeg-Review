#!/bin/bash

source source.sh

VISIBLE_GPUS=$1

TRAIN_SET="./data/uavid_image_crop_512x512/uavid_train"
VALID_SET="./data/uavid_image_crop_512x512/uavid_val"
OUTPUT_ROOT="./output"

PORT=$(rand 30000 50000)

CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS python ./main.py \
  --input_height 512 \
  --input_width 512 \
  --num_epoch 30 \
  --batch_size 8 \
  --num_workers 16 \
  --optimizer 'Adam' \
  --init_lr 1e-4 \
  --lr_gamma 0.9 \
  --momentum 0.9 \
  --weight_decay 1e-4 \
  --phase "train" \
  --seed 369 \
  --model "TransUNetR50ViTB16" \
  --dataset "UAVid" \
  --train_set "${TRAIN_SET}" \
  --valid_set "${VALID_SET}" \
  --output_root "${OUTPUT_ROOT}" \
  --resume "epoch_last.pth" \
  --init_group \
  --init_method "tcp://localhost:${PORT}"
