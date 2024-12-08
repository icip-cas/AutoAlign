ROOT=${ROOT:-"../../../autoalign"}
SRC_PATH="${ROOT}/src/autoalign/train_megatron/toolkits/sft_conv_data_preprocessing"
INPUT_JSON=${INPUT_JSON:-"${ROOT}/data/sft/dummy.json"}
DATA_TYPE=${DATA_TYPE:-"conversations"}
TOKENIZER=${TOKENIZER:-"Qwen2Tokenizer"}
SEQ_LEN=${SEQ_LEN:-4096}
OUTPUT_DATA_PREFIX=${OUTPUT_DATA_PREFIX:-"${ROOT}/data/sft/dummy"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"${ROOT}/hf_models/Qwen2.5-7B"}
cd ${SRC_PATH}

bash run_build_idxmap_sft_conv_dataset.sh \
    ${INPUT_JSON} \
    ${DATA_TYPE} \
    ${TOKENIZER} \
    ${SEQ_LEN} \
    ${OUTPUT_DATA_PREFIX} \
    ${HF_MODEL_PATH}