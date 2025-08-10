unset SETUPTOOLS_USE_DISTUTILS
export OMP_NUM_THREADS=4

autoalign-cli rl --algorithm grpo \
    --model_name_or_path ${MODEL_PATH:-"Qwen2/Qwen2-7B"} \
    --dataset_name ${DATA_PATH:-"data/dummy_rl.json"} \
    --output_dir ${OUTPUT_DIR:-"outputs/qwen2-7b"} \
    --attn_implementation flash_attention_2 \
    --log_level info \
    --logging_steps 1 \
    --logging_first_step True \
    --save_strategy ${SAVE_STRATEGY:-"steps"} \
    --save_steps ${EVAL_STEPS:-"300"} \
    --reward_funcs "xverify_reward" \
    --bf16 True \
    --torch_dtype bfloat16 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 4 \
    --learning_rate ${LR:-"1e-6"} \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs $'\'{"min_lr_rate": "0.1"}\'' \
    --max_prompt_length ${MAX_PROMPT_LENGTH:-"512"} \
    --max_completion_length ${MAX_COMPLETION_LENGTH:-"4096"} \
    --num_generations ${NUM_GENERATIONS:-"7"} \
    --temperature ${TEMPERATURE:-"0.7"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"1"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"4"} \
    --warmup_ratio ${WARMUP_RATIO:-"0.1"} \
    --do-eval False \
    --num-train-epochs ${EPOCHS:-"1"} \
    --overwrite_output_dir True \
    --use_liger_kernel True \
    --use_xverify True \
    --xverify_model_path ${XVERIFY_MODEL_PATH:-"IAAR-Shanghai/xVerify"}