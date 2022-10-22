#!/bin/bash

# PYTHONPATH=~/Basic_SISR
n_b=2
n_f=32

CUDA_VISIBLE_DEVICES=2 python test.py --data_root /home/hong/dataset/oneh_data/ \
      --input_name LR \
      --target_name HR \
      --model_type EDSR \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 10 --num_valid_image 3\
      --model_save_root /home/hong/1017_dir/sr-training/code/save_model \
      --num_batch 1 \
      --num_update_per_epoch 100  \
      --pretrained --pretrained_path naive_models/oneh_2_32/EDSR_0.pth
      # --pretrained --pretrained_path pretrained_model/2_32.pth \
      # --pretrained --pretrained_path cls_models/50cls300/6/EDSR_90.pth \


# # PYTHONPATH=~/Basic_SISR
# n_b=2
# n_f=32

# n_cls=20
# for var in 0 1 2 6 7 8 10 15
# do
# CUDA_VISIBLE_DEVICES=0 python test.py --data_root /home/hong/dataset/100data/${n_cls}cls/${var} \
#       --input_name LR \
#       --target_name HR \
#       --model_type EDSR \
#       --scale 4 --n_blocks $n_b --n_feats $n_f \
#       --use_cuda --num_epoch 10 --num_valid_image 3\
#       --model_save_root /home/hong/1017_dir/sr-training/code/save_model \
#       --num_batch 1 \
#       --num_update_per_epoch 100  \
#       --pretrained --pretrained_path naive_models/dev300/EDSR_295.pth
#       # --pretrained --pretrained_path cls_models/50cls300/6/EDSR_90.pth \
#       # --pretrained --pretrained_path pretrained_model/2_32.pth \
# done