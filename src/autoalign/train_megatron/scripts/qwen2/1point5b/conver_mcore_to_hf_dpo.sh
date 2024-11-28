export CUDA_VISIBLE_DEVICES=0
cd /share/zhangqingyu/code/auto-alignment/algorithms/megatron_dpo/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2_convertor_dpo.sh \
1.5B \
/share/zhangqingyu/checkpoint/dpo/output_mcore_qwen2_1point5_ct_tp2_pp2/checkpoint/pretrain-mcore-llama3-1-1.5B-lr-1e-5-minlr-1e-6-bs-4-gbs-32-seqlen-2048-pr-bf16-tp-2-pp-2-cp-1-ac-false-do-true-sp-ture-ti-15258-wi-0 \
/share/zhangqingyu/hf_models/Qwen2-1.5B-tp2-pp2-dpo \
2  \
2  \
1 \
fp32 \
true \
true \
/share/zhangqingyu/hf_models/Qwen2-1.5B