#!/bin/bash

TRAIN_DATA="/path/to/imagenet/train"
VAL_DATA="/path/to/imagenet/val"
ENCODER_PATH="checkpoints/256x256_diffusion_uncond.pt"

MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
WANDB_RUN_NAME="linear"
OUTPUT_ENV="./out/"$WANDB_RUN_NAME

python -m torch.distributed.launch --master_port=$((1000 + RANDOM % 9999)) --nproc_per_node=4 finetune.py \
    --data_dir $TRAIN_DATA \
    --val_data_dir $VAL_DATA \
    --epochs=28 \
    --lr=1e-2 \
    --batch_size=32 \
    --mode=freeze \
    --checkpoint_path $ENCODER_PATH \
    --num_classes 1000 \
    --output_dir $OUTPUT_ENV \
    --t_list 150 \
    --first_fw_b_list 24 \
    --second_fw_b_list -1 \
    --wandb_run_name $WANDB_RUN_NAME \
    --head_type linear \
    --pre_pool_size 2 \
    --eval_interval 7 \
    --feedback_b_list -1 \
    --use_wandb False \
    $MODEL_FLAGS $CLASSIFIER_FLAGS $SAMPLE_FLAGS