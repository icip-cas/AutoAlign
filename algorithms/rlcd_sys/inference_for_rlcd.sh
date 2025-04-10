export SOURCE_TAG=${MODEL_NAME}_cd
export CHOSEN_SOURCE_TAG=${SOURCE_TAG}_chosen
export REJECTED_SOURCE_TAG=${SOURCE_TAG}_rejected
echo "Method Name: RLCD"
autoalign-cli infer --backend "vllm" \
            --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODEL_PATH} \
            --test-file ${OUTPUT_DIR}/${OUTPUT_CHOSEN_FILE_NAME} \
            --template ${TEMPLATE_NAME} \
            --source ${CHOSEN_SOURCE_TAG}

autoalign-cli infer --backend "vllm" \
            --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODEL_PATH} \
            --test-file ${OUTPUT_DIR}/${OUTPUT_REJECTED_FILE_NAME} \
            --template ${TEMPLATE_NAME} \
            --source ${REJECTED_SOURCE_TAG}
