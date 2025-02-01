unset SETUPTOOLS_USE_DISTUTILS
export OMP_NUM_THREADS=4
autoalign-cli rm \
    --model_name_or_path ${MODEL_PATH:-"Qwen2/Qwen2-7B"} \
    --data_path ${DATA_PATH:-"data/ultra_binary.jsonl"} \
    --eval_path ${EVAL_DATA_PATH:-"data/eval"} \
    --conv_template_name ${CONV_TEMPLATE:-"chatml"} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --num_train_epochs ${EPOCH:-"1"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"1"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"4"} \
    --gradient_accumulation_steps ${GA:-"1"} \
    --eval_strategy ${EVAL_STRATEGY:-"steps"} \
    --eval_steps ${EVAL_STEPS:-"0.05"} \
    --save_strategy ${SAVE_STRATEGY:-"steps"} \
    --save_steps ${SAVE_STEPS:-"0.25"} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-"4"} \
    --learning_rate ${LR:-"5e-7"} \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --max_length ${MAX_LENGTH:-"1024"} \
    --lr_scheduler_type ${LR_SCHEDULE:-"cosine"} \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --logging_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --logging_steps 1 \
    --gradient_checkpointing False \
    --deepspeed ${DS_CONFIG:-"configs/zero2.json"}
