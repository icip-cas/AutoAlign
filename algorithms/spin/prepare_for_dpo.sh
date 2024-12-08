PROMPTS_FILE_NAME=$(basename ${PROMPTS_FILE})

export REJECTED_FILE=${OUTPUT_DIR}/${MODEL_NAME}/${REJECTED_SOURCE_TAG}_${PROMPTS_FILE_NAME}

export OUTPUT_FILE_NAME=${MODEL_NAME}_spin_${PROMPTS_FILE_NAME}

python -m autoalign.data.prepare_for_dpo --input-files ${CHOSEN_FILE} \
                                                        ${REJECTED_FILE} \
                                        --chosen-source ${CHOSEN_SOURCE_TAG} \
                                        --rejected-source ${REJECTED_SOURCE_TAG} \
                                        --output-file-path ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                                        --set-source-tag "0->golden" \
                                        --remove-system-message \
                                        --abandon-same-response
