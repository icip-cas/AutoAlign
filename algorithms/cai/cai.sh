mkdir -p ${OUTPUT_DIR}
mkdir -p ${SAVE_MODEL_DIR}
echo "Method Name: CAI"
python prepare_for_cai.py   --model-name ${MODEL_NAME} \
                            --model-path ${MODEL_PATH} \
                            --input-file ${PROMPTS_FILE} \
                            --input_helpful_file ${POSITVE_CHAT_FILE} \
                            --output-chosen ${OUTPUT_DIR}/${OUTPUT_CHOSEN_FILE_NAME} \
                            --output-rejected ${OUTPUT_DIR}/${OUTPUT_REJECTED_FILE_NAME} \
                            --output-cai ${OUTPUT_DIR}/${OUTPUT_CAI_FILE_NAME} \
                            --output-sft ${OUTPUT_DIR}/${OUTPUT_SFT_FILE_NAME}
echo "==============================prepare_for_cai.py have done=============================="

# sft
autoalign-cli sft \
    --model_name_or_path ${MODEL_PATH:-"Qwen2/Qwen2-7B"}  \
    --data_path ${OUTPUT_DIR}/${OUTPUT_SFT_FILE_NAME} \
    --conv_template_name ${CONV_TEMPLATE:-"chatml"} \
    --bf16 True \
    --output_dir "${SAVE_MODEL_DIR}${SFT_MODEL_NAME}" \
    --num_train_epochs ${EPOCH:-"1"} \
    --per_device_train_batch_size ${TRAIN_BATCH_SIZE:-"1"} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE:-"1"} \
    --gradient_accumulation_steps ${GA:-"32"} \
    --eval_strategy ${EVAL_STRATEGY:-"steps"} \
    --eval_steps ${EVAL_STEPS:-"10"} \
    --save_strategy ${SAVE_STRATEGY:-"epoch"} \
    --learning_rate ${LR:-"2e-5"} \
    --weight_decay 0.01 \
    --warmup_ratio ${WARMUP_RATIO:-0.04} \
    --lr_scheduler_type ${LR_SCHEDULE:-"cosine"} \
    --report_to ${REPORT_TO:-"tensorboard"} \
    --logging_dir "${SAVE_MODEL_DIR}/${SFT_MODEL_NAME}" \
    --logging_steps 1 \
    --model_max_length ${MAX_LENGTH:-"2048"} \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG:-"../../configs/zero2.json"} \
    --ddp_timeout 18000 \
    --lazy_preprocess ${LAZY:-"False"} \
    --eval_num ${EVAL_NUM:-"0"}

echo "==============================sft have done=============================="

python temperature_sample.py   --model-name ${MODEL_NAME} \
                               --model-path ${SAVE_MODEL_DIR}/${SFT_MODEL_NAME}/checkpoint-* \
                               --input-file ${OUTPUT_DIR}/${OUTPUT_CAI_FILE_NAME} \
                               --output-file ${OUTPUT_DIR}/${OUTPUT_DPO_FILE_NAME}

echo "==============================temperature_sample.py have done=============================="

autoalign-cli dpo --model_name_or_path ${SAVE_MODEL_DIR}/${SFT_MODEL_NAME}/checkpoint-* \
    --data_path ${OUTPUT_DIR}/${OUTPUT_DPO_FILE_NAME} \
    --conv_template_name ${CONV_TEMPLATE} \
    --bf16 True \
    --output_dir ${SAVE_MODEL_DIR}/"${DPO_MODEL_NAME}" \
    --num_train_epochs "1" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 2e-7 \
    --max_prompt_length 1024 \
    --max_length 2048 \
    --weight_decay 0.1 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --logging_dir ${SAVE_MODEL_DIR}/"${DPO_MODEL_NAME}" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed "../../configs/zero2.json"

echo "==============================dpo have done=============================="
