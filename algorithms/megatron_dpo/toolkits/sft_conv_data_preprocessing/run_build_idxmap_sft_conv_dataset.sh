#! /bin/bash
START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/Megatron-LM-240718


input_data_path=$1
json_keys=$2
tokenizer=$3
seq_len=$4
output_data_path=$5
load_dir=$6

if [ $tokenizer = "Qwen2Tokenizer" ]; then
  python build_idxmap_sft_conv_dataset.py \
  --sft_conv \
  --mask \
  --input ${input_data_path} \
  --json-keys ${json_keys} \
  --output-prefix ${output_data_path} \
  --load ${load_dir} \
  --patch-tokenizer-type Qwen2Tokenizer \
  --model-max-length ${seq_len} \
  --workers 256 \
  --chunk-size 32 \
  --extra-vocab-size 293 \
  --dataset-impl mmap 

elif [ $tokenizer = "LLama3Tokenizer" ]; then
  python build_idxmap_sft_conv_dataset.py \
  --sft_conv \
  --mask \
  --input ${input_data_path} \
  --json-keys ${json_keys} \
  --output-prefix ${output_data_path} \
  --load ${load_dir} \
  --patch-tokenizer-type LLama3Tokenizer \
  --model-max-length ${seq_len} \
  --workers 256 \
  --chunk-size 32 \
  --dataset-impl mmap
fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
