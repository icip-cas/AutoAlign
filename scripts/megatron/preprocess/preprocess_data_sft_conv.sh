export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd ../../src/autoalign/train_megatron/toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
/ciphome/zhangqingyu2023/code/auto-alignment/src/autoalign/train_megatron/data/sft/sharegpt_formatted_data-evol-gpt4.json \
conversations \
Qwen2Tokenizer \
4096 \
/ciphome/zhangqingyu2023/code/auto-alignment/src/autoalign/train_megatron/data/sft/sharegpt_formatted_data-evol-gpt4 \
/ciphome/zhangqingyu2023/hf_models/Qwen2.5-7B