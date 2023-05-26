#!/bin/bash

n_b=2
n_f=32
n_cls=20

SCRIPTDIR=$(dirname $0)
BASEDIR=$(realpath "${SCRIPTDIR}/../")
MODEL_DIR='logs'

while getopts m: opt
do
      case $opt in
            m) MODEL_DIR=$OPTARG
            ;;
      esac
done

python test.py --data_root  ${BASEDIR}/data \
      --model_type EDSR \
      --scale 4 --n_blocks $n_b --n_feats $n_f \
      --use_cuda --num_epoch 10 --num_valid_image 3\
      --num_batch 1 \
      --pretrained --pretrained_dir $MODEL_DIR\
      --cluster --cluster_num 20 --img_save


# If you download and use the "result_model", 
# change "logs" directory to "result_model"
# ex) in line 15, "--pretrained --pretrained_dir result_model\"