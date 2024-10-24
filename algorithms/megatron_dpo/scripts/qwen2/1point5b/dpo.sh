cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/example/qwen2
sh run_mcore_qwen_dpo.sh  \
dsw  \
1.5B   \
1    \
1 \
1e-5   \
1e-6   \
2048 \
2048 \
bf16  \
1   \
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
/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data/dummy_dpo_conversations_maxlen_2048   \
/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data/dummy_dpo_conversations_maxlen_2048  \
/ciphome/zhangqingyu2023/mg_models/Qwen2-1.5B-hf-to-mcore-te-tp1-pp1 \
1000000000  \
100   \
/ciphome/zhangqingyu2023/checkpoint/sft/output_mcore_qwen2_1point5_ct_tp2_pp1