#!/bin/bash

# PYTHONPATH=~/Basic_SISR
     
# n_b=4
# n_f=32

# CUDA_VISIBLE_DEVICES=1 python train.py --data_root /home/hong/dataset/piano-png/ \
#       --input_name LR \
#       --target_name HR \
#       --model_type EDSR \
#       --model_save_root /home/hong/1017_dir/sr-training/code/save_model \
#       --scale 4 --n_blocks $n_b --n_feats $n_f \
#       --use_cuda --num_epoch 10 --num_valid_image 3\
#       --num_batch 1 \
#       --num_update_per_epoch 1000  \
#       --pretrained --pretrained_path pretrained_model/4_32.pth


n_b=2
n_f=32
n_cls=20

for var in 9
do 
CUDA_VISIBLE_DEVICES=2 python train.py --data_root /home/hong/dataset/oneh_data/${n_cls}cls/${var} \
      --input_name LR \
      --target_name HR \
      --model_type EDSR \
      --model_save_root /home/hong/1017_dir/sr-training/cls_models/oneh_${n_cls}cls300/${var} \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 300 --num_valid_image 3\
      --num_batch 1 \
      --num_update_per_epoch 300  \
      --pretrained --pretrained_path pretrained_model/2_32.pth
done
