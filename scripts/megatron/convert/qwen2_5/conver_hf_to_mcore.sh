export CUDA_VISIBLE_DEVICES=7
cd ../../src/autoalign/train_megatron/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
7B \
/mnt/userdata/hf_models/qwen/Qwen2.5-7B \
/ciphome/zhangqingyu2023/mg_models/Qwen2.5-7B-hf-to-mcore-te-tp1-pp4  \
1  \
4  \
bf16 \
true \
false \