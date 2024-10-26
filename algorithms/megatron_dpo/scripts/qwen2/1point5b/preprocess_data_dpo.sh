export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd /ciphome/zhangqingyu2023/code/auto-alignment/algorithms/megatron_dpo/toolkits/dpo_data_preprocessing
bash run_build_idxmap_dpo_dataset.sh \
/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized.json \
conversations \
Qwen2Tokenizer \
2048 \
/ciphome/zhangqingyu2023/data/dpo/ultrafeedback_binarized_apply_template \
/ciphome/zhangqingyu2023/hf_models/Qwen2-1.5B