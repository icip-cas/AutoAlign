export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd /share/zhangqingyu/code/auto-alignment/algorithms/megatron_dpo/toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
/share/zhangqingyu/data/sft/sharegpt_formatted_data-evol-gpt4.json \
conversations \
Qwen2Tokenizer \
4096 \
/share/zhangqingyu/data/sft/sharegpt_formatted_data-evol-gpt4 \
/share/zhangqingyu/hf_models/Qwen2-1.5B