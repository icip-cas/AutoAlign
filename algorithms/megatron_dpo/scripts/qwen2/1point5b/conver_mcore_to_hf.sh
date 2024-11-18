export CUDA_VISIBLE_DEVICES=0
cd /share/zhangqingyu/code/auto-alignment/algorithms/megatron_dpo/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_convertor.sh \
1.5B \
/share/zhangqingyu/checkpoint/sft/output_mcore_qwen2_1point5_ct_tp2_pp2/checkpoint/sft-mcore-qwen2-1point5b-lr-1e-5-minlr-1e-6-bs-4-gbs-64-seqlen-4096-pr-bf16-tp-2-pp-2-cp-1-ac-none-do-true-sp-true-ti-10000-wi-100 \
/share/zhangqingyu/hf_models/Qwen2-1.5B-tp2-pp2-sft \
2  \
2  \
1 \
fp32 \
true \
true \
/share/zhangqingyu/hf_models/Qwen2-1.5B