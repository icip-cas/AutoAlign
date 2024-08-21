export CHOSEN_FILE=${OUTPUT_DIR}/${MODEL_NAME}/${CHOSEN_SOURCE_TAG}_${OUTPUT_CHOSEN_FILE_NAME}
export REJECTED_FILE=${OUTPUT_DIR}/${MODEL_NAME}/${REJECTED_SOURCE_TAG}_${OUTPUT_REJECTED_FILE_NAME}

PROMPTS_FILE_NAME=$(basename ${PROMPTS_FILE})

export OUTPUT_FILE_NAME=${MODEL_NAME}_rlcd_sys_${PROMPTS_FILE_NAME}

python -m autoalign.data.prepare_for_dpo --input-files ${CHOSEN_FILE} \
                                                        ${REJECTED_FILE} \
                                        --chosen-source ${CHOSEN_SOURCE_TAG} \
                                        --rejected-source ${REJECTED_SOURCE_TAG} \
                                        --output-file-path ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                                        --remove-system-message \
                                        --abandon-same-response
