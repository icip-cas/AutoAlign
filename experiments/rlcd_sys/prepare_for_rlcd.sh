mkdir -p ${OUTPUT_DIR}

python prepare_for_rlcd.py --input-file ${PROMPTS_FILE} \
                            --output-chosen ${OUTPUT_DIR}/${OUTPUT_CHOSEN_FILE_NAME} \
                            --output-rejected ${OUTPUT_DIR}/${OUTPUT_REJECTED_FILE_NAME}