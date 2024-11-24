export CUDA_VISIBLE_DEVICES=0
cd toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
1.5B \
/share/zhangqingyu/hf_models/Qwen2.5-1.5B \
/share/zhangqingyu/mg_models/Qwen2.5-1.5B-hf-to-mcore-te-tp1-pp1  \
1  \
1  \
fp32 \
true \
false \