cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/example/qwen2
sh run_mcore_qwen_dpo.sh  \
dsw  \
1.5B   \
2   \
32 \
1e-5   \
1e-6   \
2048 \
2048 \
bf16  \
2  \
2  \
1 \
1 \
ture \
true   \
true \
ture \
false   \
false \
100000  \
/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_apply_template_conversations_maxlen_2048   \
/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_apply_template_conversations_maxlen_2048  \
/ciphome/zhangqingyu2023/mg_models/Qwen2-1.5B-hf-to-mcore-te-tp2-pp2 \
1000000000  \
100   \
/ciphome/zhangqingyu2023/checkpoint/dpo/output_mcore_qwen2_1point5_ct_tp2_pp2