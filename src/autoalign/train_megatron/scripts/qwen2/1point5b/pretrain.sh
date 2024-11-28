cd /share/zhangqingyu/code/auto-alignment/algorithms/megatron_dpo/examples/qwen2
sh run_mcore_qwen.sh  \
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
2  \
1 \
1 \
true \
true   \
true \
false \
false   \
false \
100000  \
/share/zhangqingyu/data/qwen-datasets/wudao_qwenbpe_text_document   \
/share/zhangqingyu/data/qwen-datasets/wudao_qwenbpe_text_document   \
/share/zhangqingyu/mg_models/Qwen2.5-1.5B-hf-to-mcore-te-tp2-pp2 \
1000000000  \
100   \
/share/zhangqingyu/checkpoint/con-pre/output_mcore_qwen2_1point5_ct_tp2_pp2