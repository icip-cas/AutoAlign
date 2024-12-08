export CUDA_VISIBLE_DEVICES=7
cd ../../src/autoalign/train_megatron/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
7B \
/ciphome/zhangqingyu2023/checkpoint/sft/Qwen2.5-7B-hf-to-mcore-te-tp2-pp1-2m/checkpoint/sft-mcore-qwen2_5-7B-lr-5e-6-minlr-0.0-bs-4-gbs-512-seqlen-4096-pr-bf16-tp-2-pp-1-cp-1-ac-none-do-true-sp-false-ti-10000-wi- \
/ciphome/zhangqingyu2023/hf_models/Qwen2.5-7B-tp2-pp1-sft-2m-mg \
2  \
1  \
fp32 \
true \
true \
/ciphome/zhangqingyu2023/hf_models/Qwen2.5-7B-tp2-pp1-sft-checkpoint10000-infinite_9m