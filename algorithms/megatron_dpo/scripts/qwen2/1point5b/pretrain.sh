cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/example/qwen2
sh run_mcore_pretrain_1point5b.sh  \
dsw  \
1.5B   \
2    \
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
false \
false   \
false \
100000  \
/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data/wudao_qwenbpe_text_document   \
/ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/data/wudao_qwenbpe_text_document   \
/ciphome/zhangqingyu2023/mg_models/Qwen2-1.5B-hf-to-mcore-te-tp2-pp1 \
1000000000  \
100   \
/ciphome/zhangqingyu2023/checkpoint/con-pre/output_mcore_qwen2_1point5_ct_tp2_pp1