export CUDA_VISIBLE_DEVICES=0
cd toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
1.5B \
/share/zhangqingyu/checkpoint/sft/output_mcore_qwen2_5_1point5_ct_tp1_pp1/checkpoint/sft-mcore-qwen2_5-1point5b-lr-1e-5-minlr-1e-6-bs-1-gbs-16-seqlen-8192-pr-bf16-tp-1-pp-1-cp-1-ac-none-do-true-sp-true-ti-10000-wi-10 \
/share/zhangqingyu/hf_models/Qwen2.5-1.5B-tp1-pp1-sft \
1  \
1  \
fp32 \
true \
true \
/share/zhangqingyu/hf_models/Qwen2.5-1.5B