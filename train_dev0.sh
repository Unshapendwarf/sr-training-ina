#!/bin/bash

# PYTHONPATH=~/Basic_SISR
     
n_b=2
n_f=32

CUDA_VISIBLE_DEVICES=2 python train.py --data_root /home/hong/dataset/oneh_data/ \
      --input_name LR \
      --target_name HR \
      --model_type EDSR \
      --model_save_root /home/hong/1017_dir/sr-training/naive_models/oneh_2_32 \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 300 --num_valid_image 3\
      --num_batch 1 \
      --num_update_per_epoch 300  \
      --pretrained --pretrained_path pretrained_model/${n_b}_${n_f}.pth \


# n_b=4
# n_f=32
# n_cls=32

# for var in {0..31}
# do 
# echo /home/hong/dataset/piano-clustered/${n_cls}cls/${var}
# CUDA_VISIBLE_DEVICES=1 python train.py --data_root /home/hong/dataset/piano-clustered/${n_cls}cls/${var} \
#       --input_name LR \
#       --target_name HR \
#       --model_type EDSR \
#       --model_save_root /home/hong/1017_dir/sr-training/code/cls_models/${n_cls}cls/${var} \
#       --scale 4 --n_blocks $n_b --n_feats $n_f \
#       --use_cuda --num_epoch 20 --num_valid_image 3\
#       --num_batch 1 \
#       --num_update_per_epoch 1500  \
#       --pretrained --pretrained_path pretrained_model/4_32.pth
# done