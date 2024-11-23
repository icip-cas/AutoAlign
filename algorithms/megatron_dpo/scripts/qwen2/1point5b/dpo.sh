cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/examples/qwen2
sh run_mcore_qwen_dpo.sh  \
dsw  \
1.5B   \
4   \
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
false \
true   \
true \
ture \
false   \
false \
100000  \
/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_conversations_maxlen_2048   \
/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_conversations_maxlen_2048  \
/ciphome/zhangqingyu2023/mg_models/Qwen2.5-1.5B-hf-to-mcore-te-tp2-pp2 \
1000000000  \
100   \
/ciphome/zhangqingyu2023/mg_models/Qwen2.5-1.5B-hf-to-mcore-te-tp2-pp2