autoalign-cli infer --backend "vllm" \
            --model-name ${STRONG_MODEL_NAME} \
            --model-path ${STRONG_MODEL_PATH} \
            --test-file ${PROMPTS_FILE} \
            --template ${TEMPLATE_NAME} \
            --source ${STRONG_MODEL_NAME} \

autoalign-cli infer --backend "vllm" \
            --model-name ${WEAK_MODEL_NAME} \
            --model-path ${WEAK_MODEL_PATH} \
            --test-file ${PROMPTS_FILE} \
            --template ${TEMPLATE_NAME} \
            --source ${WEAK_MODEL_NAME} \