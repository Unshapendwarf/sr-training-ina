#!/bin/bash

n_b=2
n_f=32
n_cls=20

SCRIPTDIR=$(dirname $0)
BASEDIR=$(realpath "${SCRIPTDIR}/../")

CUDA_VISIBLE_DEVICES=2 python train.py --data_root ${BASEDIR}/data/ \
      --model_type EDSR \
      --log_dir ${BASEDIR}/logs \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 3 --num_valid_image 5\
      --num_batch 1 \
      --num_update_per_epoch 30  \
      --pretrained --pretrained_path pretrained_model/2_32.pth\
      --cluster