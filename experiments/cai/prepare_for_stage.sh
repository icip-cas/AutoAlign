if [ ${STAGE} -eq 0 ]; then
    mkdir -p ${OUTPUT_DIR}
    python prepare_for_cai.py --input-file ${PROMPTS_FILE} \
                                --stage 0 \
                                --output ${OUTPUT_DIR}/${OUTPUT_FILE_NAME}
elif [ ${STAGE} -eq 1 ]; then
    mkdir -p ${OUTPUT_DIR}
    export SOURCE_TAG=${MODEL_NAME}_cai_stage${STAGE}

    python prepare_for_cai.py --input-file ${C_PROMPTS_FILE} \
                                --stage ${STAGE} \
                                --output ${OUTPUT_DIR}/${OUTPUT_FILE_NAME}

    autoalign-cli infer --backend "vllm" \
                --model-name ${MODEL_NAME} \
                --model-path ${SAVED_MODEL_PATH} \
                --test-file ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                --template ${TEMPLATE_NAME} \
                --source ${SOURCE_TAG}
else
    python prepare_for_cai.py --input-file ${C_PROMPTS_FILE} \
                            --stage ${STAGE} \
                            --output ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                            --last-stage-output ${LAST_STAGE_OUTPUT}

    autoalign-cli infer --backend "vllm" \
                --model-name ${MODEL_NAME} \
                --model-path ${SAVED_MODEL_PATH} \
                --test-file ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                --template ${TEMPLATE_NAME} \
                --source ${SOURCE_TAG}
fi
