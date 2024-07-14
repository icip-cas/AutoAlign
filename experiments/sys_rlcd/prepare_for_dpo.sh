CHOSEN_FILE=${OUTPUT_DIR}/${CHOSEN_SOURCE_TAG}_${PROMPTS_FILE_NAME}
REJECTED_FILE=${OUTPUT_DIR}/${REJECTED_SOURCE_TAG}_${PROMPTS_FILE_NAME}
OUTPUT_FILE_NAME=${MODEL_NAME}_rlcdsys_${PROMPTS_FILE_NAME}

python ./src/autoalign/data/prepare_for_dpo.py --input-files ${CHOSEN_FILE} \
                                                        ${REJECTED_FILE} \
                                            --chosen-source ${CHOSEN_SOURCE_TAG} \
                                            --rejected-source ${REJECTED_SOURCE_TAG} \
                                            --output-file-path ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                                            --abandon-same-response