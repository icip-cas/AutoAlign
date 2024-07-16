export CHOSEN_FILE=${OUTPUT_DIR}/${STONG_MODEL_NAME}/${CHOSEN_SOURCE_TAG}_${OUTPUT_CHOSEN_FILE_NAME}
export REJECTED_FILE=${OUTPUT_DIR}/${WEAK_MODEL_NAME}/${REJECTED_SOURCE_TAG}_${OUTPUT_REJECTED_FILE_NAME}

PROMPTS_FILE_NAME=$(basename ${PROMPTS_FILE})

export OUTPUT_FILE_NAME=${STONG_MODEL_NAME}_pcon_${PROMPTS_FILE_NAME}

python -m autoalign.data.prepare_for_dpo --input-files ${CHOSEN_FILE} \
                                                        ${REJECTED_FILE} \
                                        --chosen-source ${STONG_MODEL_NAME} \
                                        --rejected-source ${WEAK_MODEL_NAME} \
                                        --output-file-path ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                                        --abandon-same-response