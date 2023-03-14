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

BASEDIR=$(dirname $0)

for var in {0..20..1}
do 
echo ${var}
CUDA_VISIBLE_DEVICES=2 python train.py --data_root ${BASEDIR}/data/${n_cls}cls/${var} \
      --input_name LR \
      --target_name HR \
      --model_type EDSR \
      --model_save_root ${BASEDIR}/logs \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 3 --num_valid_image 3\
      --num_batch 1 \
      --num_update_per_epoch 10  \
      --pretrained --pretrained_path pretrained_model/2_32.pth
done
