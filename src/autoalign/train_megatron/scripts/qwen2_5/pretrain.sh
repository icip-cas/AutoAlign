cd examples/qwen2_5
sh run_mcore_qwen.sh  \
dsw  \
1.5B   \
1    \
1 \
1e-5   \
1e-6   \
4096  \
4096  \
bf16  \
2   \
2  \
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
/share/zhangqingyu/mg_models/Qwen2.5-1.5B-hf-to-mcore-te-tp2-pp2  \
10000000000  \
1000   \
/share/zhangqingyu/checkpoint/pretrains/Qwen2.5-1.5B-hf-to-mcore-te-tp2-pp2