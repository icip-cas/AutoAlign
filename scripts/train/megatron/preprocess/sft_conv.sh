START_TIME=$SECONDS
ROOT=${ROOT:-"../../auto-alignment"}
INPUT_JSON=${INPUT_JSON:-"${ROOT}/data/sft/dummy.json"}
DATA_TYPE=${DATA_TYPE:-"conversations"}
TOKENIZER=${TOKENIZER:-"Qwen2Tokenizer"}
SEQ_LEN=${SEQ_LEN:-4096}
OUTPUT_DATA_PREFIX=${OUTPUT_DATA_PREFIX:-"${ROOT}/data/sft/dummy"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"${ROOT}/hf_models/Qwen2.5-7B"}
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-421}
TEMPLATE=${TEMPLATE:-"chatml-idsys"}

python toolkits/sft/preprocess.py.py \
  --sft_conv \
  --mask \
  --input ${INPUT_JSON} \
  --json-keys ${DATA_TYPE} \
  --output-prefix ${OUTPUT_DATA_PREFIX} \
  --load ${HF_MODEL_PATH} \
  --patch-tokenizer-type Qwen2Tokenizer \
  --model-max-length ${SEQ_LEN} \
  --workers 256 \
  --chunk-size 32 \
  --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
  --dataset-impl mmap \
  --template ${TEMPLATE}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
