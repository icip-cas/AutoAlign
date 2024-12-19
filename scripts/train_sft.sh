unset SETUPTOOLS_USE_DISTUTILS
export OMP_NUM_THREADS=4
autoalign-cli sft \
    --model_name_or_path ${MODEL_PATH:-"Qwen2/Qwen2-7B"}  \
    --data_path ${DATA_PATH:-"data/dummy_sft.json"} \
    --conv_template_name ${CONV_TEMPLATE:-"chatml"} \
    --bf16 True \
    --output_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --num_train_epochs ${EPOCH:-"3"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"4"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"4"} \
    --gradient_accumulation_steps ${GA:-"1"} \
    --eval_strategy ${EVAL_STRATEGY:-"no"} \
    --eval_steps ${EVAL_STEPS:-"1500"} \
    --save_strategy ${SAVE_STRATEGY:-"epoch"} \
    --learning_rate ${LR:-"2e-5"} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO:-0.04} \
    --lr_scheduler_type ${LR_SCHEDULE:-"cosine"} \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --logging_dir ${OUTPUT_DIR:-"models/qwen2-7b"} \
    --logging_steps 1 \
    --model_max_length ${MAX_LENGTH:-"4096"} \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG:-"configs/zero3.json"} \
    --ddp_timeout 18000 \
    --lazy_preprocess ${LAZY:-"False"} \
    --eval_num ${EVAL_NUM:-"0"}
