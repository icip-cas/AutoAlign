export CUDA_VISIBLE_DEVICES=0
cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_convertor.sh \
1.5B \
/share/zhangqingyu/hf_models/Qwen2.5-1.5B \
/share/zhangqingyu/mg_models/Qwen2.5-1.5B-hf-to-mcore-te-tp2-pp2  \
2  \
2  \
1 \
fp32 \
true \
false 