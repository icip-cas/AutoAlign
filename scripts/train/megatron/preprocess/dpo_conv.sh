#! /bin/bash
START_TIME=$SECONDS

ROOT=${ROOT:-"AutoAlign"}
INPUT_JSON=${INPUT_JSON:-"${ROOT}/data/dummy_sft.json"}
DATA_TYPE=${DATA_TYPE:-"conversations"}
TOKENIZER=${TOKENIZER:-"Qwen2Tokenizer"}
SEQ_LEN=${SEQ_LEN:-4096}
OUTPUT_DATA_PREFIX=${OUTPUT_DATA_PREFIX:-"${ROOT}/data/megatron/dummy_dpo"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"${ROOT}/hf_models/Qwen2.5-7B"}
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-421}
TEMPLATE=${TEMPLATE:-"chatml-idsys"}

python ${ROOT}/src/megatron_autoalign/toolkits/dpo/preprocess.py \
  --dpo \
  --mask \
  --input ${input_data_path} \
  --json-keys ${json_keys} \
  --output-prefix ${output_data_path} \
  --load ${load_dir} \
  --patch-tokenizer-type Qwen2Tokenizer \
  --model-max-length ${seq_len} \
  --workers 256 \
  --chunk-size 32 \
  --extra-vocab-size ${extra_vocab_size} \
  --dataset-impl mmap \
  --template ${template}

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
