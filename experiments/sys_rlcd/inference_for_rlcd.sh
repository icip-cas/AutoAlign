export SOURCE_TAG=${MODEL_NAME}_cd
export CHOSEN_SOURCE_TAG=${SOURCE_TAG}_chosen
export REJECTED_SOURCE_TAG=${SOURCE_TAG}_rejected

autoalign-cli infer --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODLE_PATH} \
            --test-file ${OUTPUT_DATA_DIR}/${OUTPUT_REJECTED_FILE_NAME} \
            --template ${TEMPLATE} \
            --source "${CHOSEN_SOURCE_TAG}" \

autoalign-cli infer --model-name ${MODEL_NAME} \
            --model-path ${SAVED_MODLE_PATH} \
            --test-file ${OUTPUT_DATA_DIR}/${OUTPUT_REJECTED_FILE_NAME} \
            --template ${TEMPLATE} \
            --source ${REJECTED_SOURCE_TAG} \