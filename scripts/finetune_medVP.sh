# !/bin/bash
# export NCCL_P2P_DISABLE="1" 
# export NCCL_IB_DISABLE="1"
PROMPT_VERSION=llava_v1
model_name_or_path=/path/to/base/model_name_or_path
CUDA=0,1,2,3
output_dir=/path/to/output/dir
img_path=/path/to/image/folder
json_path=/path/to/train/json_file

cd llava/train || exit
deepspeed --include localhost:$CUDA --master_port=$RANDOM finetune_MedVP.py \
    --deepspeed ../../scripts/zero3_offload.json \
    --model_name_or_path $model_name_or_path \
    --version $PROMPT_VERSION \
    --data_path $json_path \
    --image_folder $img_path \
    --vision_tower clip_4layers_336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 15 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

