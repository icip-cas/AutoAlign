#! /bin/bash
START_TIME=$SECONDS

cd toolkits/dpo_data_preprocessing
input_data_path=/ciphome/zhangqingyu2023/code/auto-alignment/data/dummy_dpo.json
json_keys=conversations
tokenizer=Qwen2Tokenizer
seq_len=2048
output_data_path=/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data
load_dir=/ciphome/zhangqingyu2023/hf_models/Qwen2-1.5B

if [ $tokenizer = "Qwen2Tokenizer" ]; then
  python build_idxmap_dpo_dataset.py \
  --input ${input_data_path} \
  --json-keys ${json_keys} \
  --output-prefix ${output_data_path} \
  --tokenizer-name-or-path ${load_dir} \
  --patch-tokenizer-type Qwen2Tokenizer \
  --model-max-length ${seq_len} \
  --workers 256 \
  --chunk-size 32 \
  --dataset-impl mmap 

elif [ $tokenizer = "LLama3Tokenizer" ]; then
  python build_idxmap_dpo_dataset.py \
  --input ${input_data_path} \
  --json-keys ${json_keys} \
  --output-prefix ${output_data_path} \
  --tokenizer-name-or-path ${load_dir} \
  --patch-tokenizer-type LLama3Tokenizer \
  --model-max-length ${seq_len} \
  --workers 256 \
  --chunk-size 32 \
  --dataset-impl mmap
fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
