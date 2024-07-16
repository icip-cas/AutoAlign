export SOURCE_TAG=${MODEL_NAME}_cd
export CHOSEN_SOURCE_TAG=${SOURCE_TAG}_chosen
export REJECTED_SOURCE_TAG=${SOURCE_TAG}_rejected

autoalign-cli infer --backend "vllm" \
            --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODEL_PATH} \
            --test-file ${OUTPUT_DIR}/${OUTPUT_REJECTED_FILE_NAME} \
            --template ${TEMPLATE_NAME} \
            --source "${CHOSEN_SOURCE_TAG}" \
            --debug-mode

autoalign-cli infer --backend "vllm" \
            --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODEL_PATH} \
            --test-file ${OUTPUT_DIR}/${OUTPUT_REJECTED_FILE_NAME} \
            --template ${TEMPLATE_NAME} \
            --source "${REJECTED_SOURCE_TAG}" \
            --debug-mode