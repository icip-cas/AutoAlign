export CUDA_VISIBLE_DEVICES=7
cd ../../src/autoalign/train_megatron/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
7B \
/ciphome/zhangqingyu2023/mg_models/Qwen2.5-7B-hf-to-mcore-te-tp2-pp2 \
/ciphome/zhangqingyu2023/hf_models/Qwen2.5-7B-mcore-to-hf-tp2-pp2-test \
2  \
2  \
fp32 \
true \
true \
/mnt/userdata/hf_models/qwen/Qwen2.5-7B