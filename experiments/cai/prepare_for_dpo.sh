PROMPTS_FILE_NAME=$(basename ${PROMPTS_FILE})
export OUTPUT_FILE_NAME=${MODEL_NAME}_cai_${PROMPTS_FILE_NAME}

export CHOSEN_FILE=${OUTPUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_cai_ultrafeedback_ata_chosen.json
export REJECTED_FILE=${OUTPUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_cai_ultrafeedback_ata_rejected.json
export SOURCE_TAG=${MODEL_NAME}_cai
export CHOSEN_SOURCE_TAG=${SOURCE_TAG}_chosen
export REJECTED_SOURCE_TAG=${SOURCE_TAG}_rejected

python delete_history.py --raw-file ${C_PROMPTS_FILE} \
                         --input-file ${OUTPUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_cai_stage1_ultrafeedback_ata_stage1.json \
                         --output ${CHOSEN_FILE} \
                         --source-tag ${CHOSEN_SOURCE_TAG}

python delete_history.py --raw-file ${C_PROMPTS_FILE} \
                         --input-file ${OUTPUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_cai_stage3_ultrafeedback_ata_stage3.json \
                         --output ${REJECTED_FILE} \
                         --source-tag ${REJECTED_SOURCE_TAG}

python -m autoalign.data.prepare_for_dpo --input-files ${CHOSEN_FILE} \
                                                        ${REJECTED_FILE} \
                                        --chosen-source ${CHOSEN_SOURCE_TAG} \
                                        --rejected-source ${REJECTED_SOURCE_TAG} \
                                        --output-file-path ${OUTPUT_DIR}/${OUTPUT_FILE_NAME} \
                                        --remove-system-message \
                                        --abandon-same-response
