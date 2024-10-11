#!/bin/bash

source source.sh

VISIBLE_GPUS=$1

TEST_SET="./data/uavid_image_crop_512x512/uavid_test"
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
  --phase "test" \
  --seed 369 \
  --model "SegNetVGG16" "FCN8s" "FCN16s" "PSPNetVGG16" "PSPNetResNet50" "PSPNetResNet101" "DeepLabV3ResNet50" "DeepLabV3ResNet101" "DeepLabV3PlusResNet50" "DeepLabV3PlusResNet101" "UNetResNet50" "UNetResNet101" "UNetPlusPlusResNet50" "UNetPlusPlusResNet101" "HRNetV2W48" "BiSeNetV1" \
  --dataset "UAVid" \
  --test_set "${TEST_SET}" \
  --output_root "${OUTPUT_ROOT}" \
  --resume "best_mIoU.pth" \
  --init_group \
  --init_method "tcp://localhost:${PORT}"

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
  --phase "test" \
  --seed 369 \
  --model "SegNetVGG16" "FCN8s" "FCN16s" "PSPNetVGG16" "PSPNetResNet50" "PSPNetResNet101" "DeepLabV3ResNet50" "DeepLabV3ResNet101" "DeepLabV3PlusResNet50" "DeepLabV3PlusResNet101" "UNetResNet50" "UNetResNet101" "UNetPlusPlusResNet50" "UNetPlusPlusResNet101" "HRNetV2W48" "BiSeNetV1" \
  --densecrf \
  --dataset "UAVid" \
  --test_set "${TEST_SET}" \
  --output_root "${OUTPUT_ROOT}" \
  --resume "best_mIoU.pth" \
  --init_group \
  --init_method "tcp://localhost:${PORT}"

for fs in {3..13..2} ;
  do 
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
  --phase "test" \
  --seed 369 \
  --model "SegNetVGG16" "FCN8s" "FCN16s" "PSPNetVGG16" "PSPNetResNet50" "PSPNetResNet101" "DeepLabV3ResNet50" "DeepLabV3ResNet101" "DeepLabV3PlusResNet50" "DeepLabV3PlusResNet101" "UNetResNet50" "UNetResNet101" "UNetPlusPlusResNet50" "UNetPlusPlusResNet101" "HRNetV2W48" "BiSeNetV1" \
  --convcrf \
  --convcrf_fsize ${fs} \
  --dataset "UAVid" \
  --test_set "${TEST_SET}" \
  --output_root "${OUTPUT_ROOT}" \
  --resume "best_mIoU.pth" \
  --init_group \
  --init_method "tcp://localhost:${PORT}"
done