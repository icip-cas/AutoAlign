export CUDA_VISIBLE_DEVICES=0
cd toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
7B \
/mnt/userdata/hf_models/qwen/Qwen2.5-7B \
/ciphome/zhangqingyu2023/mg_models/Qwen2.5-7B-hf-to-mcore-te-tp2-pp2  \
2  \
2  \
bf16 \
true \
false \