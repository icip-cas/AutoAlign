export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
/share/zhangqingyu/data/sft/sharegpt_formatted_data-evol-gpt4.json \
conversations \
Qwen2Tokenizer \
8192 \
/share/zhangqingyu/data/sft/sharegpt_formatted_data-evol-gpt4 \
/share/zhangqingyu/hf_models/Qwen2.5-1.5B