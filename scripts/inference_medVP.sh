#!/bin/bash
CUDA="0"
GPU=1

cd inference || exit
checkpoint=${checkpoints[$ep]}
model_path=/path/to/finetuned/model
answer_file=/path/to/output/jsonl_file
CUDA_VISIBLE_DEVICES=$CUDA python -m torch.distributed.run --nproc_per_node=$GPU --master_port=$RANDOM \
    ./inference_closed-open_form.py \
    --model-path $model_path \
    --question-file /path/to/test/json_file \
    --image-folder /path/to/img_folder \
    --answers-file $answer_file \
    --temperature 0

