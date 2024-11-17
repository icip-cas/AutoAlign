export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd /share/zhangqingyu/code/auto-alignment/algorithms/megatron_dpo/toolkits/dpo_data_preprocessing
bash run_build_idxmap_dpo_dataset.sh \
/share/zhangqingyu/data/dpo/ultrafeedback_binarized.json \
conversations \
Qwen2Tokenizer \
2048 \
/share/zhangqingyu/data/dpo/ultrafeedback_binarized \
/share/zhangqingyu/hf_models/Qwen2-1.5B