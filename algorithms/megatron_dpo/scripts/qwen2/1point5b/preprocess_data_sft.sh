cd /ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/toolkits/sft_data_preprocessing
bash run_build_idxmap_sft_dataset.sh \
/ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/data/qwen-datasets/qwen_sft.json \
Qwen2Tokenizer \
2048 \
/ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/data/qwen-datasets/mmap_qwen2_sft_datasets \
/ciphome/zhangqingyu2023/hf_models/Qwen2-1.5B