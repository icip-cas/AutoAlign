export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data/sft/sharegpt_formatted_data-evol-gpt4.json \
conversations \
Qwen2Tokenizer \
8192 \
/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data/sft/sharegpt_formatted_data-evol-gpt4 \
/mnt/userdata/hf_models/qwen/Qwen2.5-7B