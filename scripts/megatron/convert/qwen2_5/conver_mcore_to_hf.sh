export CUDA_VISIBLE_DEVICES=7
cd ../../src/autoalign/train_megatron/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
7B \
checkpoint/sft \
hf_models/Qwen2.5-7B-tp2-pp1-sft \
2  \
1  \
fp32 \
true \
true \
Qwen2.5-7B