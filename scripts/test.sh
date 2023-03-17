#!/bin/bash

n_b=2
n_f=32
n_cls=20

SCRIPTDIR=$(dirname $0)
BASEDIR=$(realpath "${SCRIPTDIR}/../")

CUDA_VISIBLE_DEVICES=2 python test.py --data_root  ${BASEDIR}/data \
      --model_type EDSR \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 10 --num_valid_image 3\
      --num_batch 1 \
      --pretrained --pretrained_dir logs\
      --cluster --cluster_num 20 --img_save