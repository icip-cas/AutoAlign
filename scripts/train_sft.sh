set -x
N_GPUS=${N_GPUS:-$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)}
torchrun --nproc_per_node="${N_GPUS:-"8"}" \
    --nnodes="${WORLD_SIZE:-"1"}" \
    --node_rank="${RANK:-"0"}" \
    --master_port="${MASTER_PORT:-"2333"}" src/autoalign/train/sft.py \
    --model_name_or_path ${MODEL_PATH:-"Qwen2/Qwen2-7B"}  \
    --data_path ${DATA_PATH:-"data/dummy_sft.json"} \
    --conv_template_name ${CONV_TEMPLATE:-"qwen-7b-chat"} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --num_train_epochs ${EPOCH:-"3"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"1"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"4"} \
    --gradient_accumulation_steps ${GA:-"4"} \
    --evaluation_strategy ${EVAL_STRATEGY:-"no"} \
    --eval_steps ${EVAL_STEPS:-"15000"} \
    --save_strategy ${SAVE_STRATEGY:-"epoch"} \
    --save_steps ${SAVE_STEPS:-"400"} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-"100"} \
    --learning_rate ${LR:-"2e-5"} \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type ${LR_SCHEDULE:-"cosine"} \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --logging_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --logging_steps 1 \
    --model_max_length ${MAX_LENGTH:-"4096"} \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG:-"configs/zero3.json"}
