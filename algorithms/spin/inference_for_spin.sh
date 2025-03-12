echo "Method Name: SPIN"
autoalign-cli infer --backend "vllm" \
            --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODEL_PATH} \
            --test-file ${PROMPTS_FILE} \
            --template ${TEMPLATE_NAME} \
            --source ${REJECTED_SOURCE_TAG}
