export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd ../../src/autoalign/train_megatron/toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
auto-alignment/src/autoalign/train_megatron/data/sft/sharegpt_formatted_data-evol-gpt4.json \
conversations \
Qwen2Tokenizer \
4096 \
auto-alignment/src/autoalign/train_megatron/data/sft/sharegpt_formatted_data-evol-gpt4 \
/Qwen2.5-7B