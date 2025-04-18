unset SETUPTOOLS_USE_DISTUTILS
export OMP_NUM_THREADS=4
autoalign-cli dpo \
    --model_name_or_path ${MODEL_PATH:-"Qwen2/Qwen2-7B"}  \
    --data_path ${DATA_PATH:-"data/dummy_dpo.json"} \
    --conv_template_name ${CONV_TEMPLATE:-"chatml"} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --num_train_epochs ${EPOCH:-"3"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"1"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"4"} \
    --gradient_accumulation_steps ${GA:-"1"} \
    --eval_strategy ${EVAL_STRATEGY:-"no"} \
    --eval_steps ${EVAL_STEPS:-"15000"} \
    --save_strategy ${SAVE_STRATEGY:-"epoch"} \
    --save_steps ${SAVE_STEPS:-"400"} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-"100"} \
    --learning_rate ${LR:-"5e-7"} \
    --beta ${BETA:-0.1} \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type ${LR_SCHEDULE:-"cosine"} \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --logging_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG:-"configs/zero3.json"}
