cd /ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/toolkits/dpo_data_preprocessing
bash run_build_idxmap_dpo_dataset.sh \
/ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/data/Anthropic--hh-rlhf/helpful-base/train_triple.json \
Qwen2Tokenizer \
2048 \
/ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/data/qwen-datasets/mmap_qwen2_dpo_datasets \
/ciphome/zhangqingyu2023/hf_models/Qwen2-1.5B