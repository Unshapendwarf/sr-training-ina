#!/bin/bash

PYTHONPATH=~/Basic_SISR
n_b=4
n_f=32

python train.py --data_root /ssd/URP_DS/final_eval \
      --input_name LR_tmp \
      --model_type RCAN \
      --model_save_root /ssd/URP_DS/final_eval/code/save_model \
      --target_name HR_tmp \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 2 --num_valid_image 3\
      --num_batch 1 \
      --num_update_per_epoch 100  \
     

