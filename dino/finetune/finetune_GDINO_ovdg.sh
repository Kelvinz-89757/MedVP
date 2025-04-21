#!/bin/bash
cd /path/to/mmdetection || exit

CUDA_VISIBLE_DEVICE=0 ./tools/dist_train.sh \
    ./grounding_dino_swin-t_pretrain_obj365.py \
    1 \
    --work-dir /path/to/store/outputs

