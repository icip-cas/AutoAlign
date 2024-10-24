cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/example/qwen2
sh run_mcore_qwen_dpo.sh  \
dsw  \
1.5B   \
1    \
8 \
1e-5   \
1e-6   \
4096  \
4096 \
bf16  \
2   \
1  \
1 \
1 \
true \
true   \
true \
ture \
false   \
false \
100000  \
/ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/data/qwen-datasets/mmap_qwen2_sft_datasets_text_document   \
/ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/data/qwen-datasets/mmap_qwen2_sft_datasets_text_document   \
/ciphome/zhangqingyu2023/mg_models/Qwen2-1.5B-hf-to-mcore-te-tp2-pp1 \
1000000000  \
100   \
/ciphome/zhangqingyu2023/checkpoint/sft/output_mcore_qwen2_1point5_ct_tp2_pp1