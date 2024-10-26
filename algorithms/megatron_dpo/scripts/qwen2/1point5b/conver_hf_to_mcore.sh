export CUDA_VISIBLE_DEVICES=2
cd /ciphome/zhangqingyu2023/code/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_convertor.sh \
1.5B \
/ciphome/zhangqingyu2023/hf_models/Qwen2-1.5B \
/ciphome/zhangqingyu2023/mg_models/Qwen2-1.5B-hf-to-mcore-te-tp2-pp2  \
2  \
2  \
1 \
fp32 \
true \
false 