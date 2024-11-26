export CUDA_VISIBLE_DEVICES=0
cd toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
7B \
/ciphome/zhangqingyu2023/mg_models/Qwen2.5-7B-hf-to-mcore-te-tp2-pp2/checkpoint/sft-mcore-qwen2_5-7B-lr-1e-5-minlr-1e-6-bs-2-gbs-16-seqlen-8192-pr-bf16-tp-2-pp-2-cp-1-ac-none-do-true-sp-true-ti-10000-wi-10 \
/ciphome/zhangqingyu2023/hf_models/Qwen2.5-7B-tp2-pp2-sft-checkpoint20 \
2  \
2  \
fp32 \
true \
true \
/mnt/userdata/hf_models/qwen/Qwen2.5-7B