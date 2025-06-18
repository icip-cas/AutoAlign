#! /bin/bash
START_TIME=$SECONDS

INPUT_JSON=${INPUT_JSON:-"./data/dummy_dpo_1024.json"}
DATA_TYPE=${DATA_TYPE:-"conversations"}
TOKENIZER=${TOKENIZER:-"Qwen2Tokenizer"}
SEQ_LEN=${SEQ_LEN:-4096}
OUTPUT_DATA_PREFIX=${OUTPUT_DATA_PREFIX:-"./data/dummy_dpo_mg"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-293}
TEMPLATE=${TEMPLATE:-"chatml-idsys"}

python -m autoalign_megatron.toolkits.dpo.preprocess \
  --dpo \
  --mask \
  --input ${INPUT_JSON} \
  --json-keys ${DATA_TYPE} \
  --output-prefix ${OUTPUT_DATA_PREFIX} \
  --load ${HF_MODEL_PATH} \
  --patch-tokenizer-type Qwen2Tokenizer \
  --model-max-length ${SEQ_LEN} \
  --workers 1 \
  --chunk-size 32 \
  --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
  --dataset-impl mmap \
  --template ${TEMPLATE} \

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
