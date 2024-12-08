CHOSEN_SOURCE_TAG=${STRONG_MODEL_NAME}
REJECTED_SOURCE_TAG=${WEAK_MODEL_NAME}

PROMPTS_FILE_NAME=$(basename ${PROMPTS_FILE})

export CHOSEN_FILE=${OUTPUT_DIR}/${STRONG_MODEL_NAME}/${CHOSEN_SOURCE_TAG}_${PROMPTS_FILE_NAME}
export REJECTED_FILE=${OUTPUT_DIR}/${WEAK_MODEL_NAME}/${REJECTED_SOURCE_TAG}_${PROMPTS_FILE_NAME}

export OUTPUT_FILE_NAME=${STRONG_MODEL_NAME}_pcon_${PROMPTS_FILE_NAME}

python -m autoalign.data.prepare_for_dpo --input-files ${CHOSEN_FILE} \
                                                        ${REJECTED_FILE} \
                                        --chosen-source ${STRONG_MODEL_NAME} \
                                        --rejected-source ${WEAK_MODEL_NAME} \
                                        --output-file-path ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                                        --abandon-same-response
