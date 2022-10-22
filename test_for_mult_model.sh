#!/bin/bash

# PYTHONPATH=~/Basic_SISR
n_b=2
n_f=32
# device="cls_models/20cls300/6"

# CUDA_VISIBLE_DEVICES=0 python test_mul.py --data_root /home/hong/dataset/100data/20cls/6 \
#       --input_name LR \
#       --target_name HR \
#       --model_type EDSR \
#       --scale 4 --n_blocks $n_b --n_feats $n_f \
#       --use_cuda --num_epoch 10 --num_valid_image 3\
#       --num_batch 1 \
#       --num_update_per_epoch 100  \
#       --pretrained --pretrained_path pretrained_model/2_32.pth \
#       --dev_path ${device} \
#       --model_save_root /home/hong/1017_dir/sr-training/code/save_model

n_cls=20
for var in {0..19}
do

# echo "${n_cls}cls/${var}" >> cls_models/${n_cls}cls/log/EDSR.log
CUDA_VISIBLE_DEVICES=0 python test_mul.py --data_root /home/hong/dataset/oneh_data/${n_cls}cls/${var} \
      --input_name LR \
      --target_name HR \
      --model_type EDSR \
      --model_save_root /home/hong/1017_dir/sr-training/code/save_model \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 10 --num_valid_image 3\
      --num_batch 1 \
      --num_update_per_epoch 100  \
      --dev_path cls_models/oneh_20cls300/${var} \
      --pretrained --pretrained_path pretrained_model/2_32.pth
done