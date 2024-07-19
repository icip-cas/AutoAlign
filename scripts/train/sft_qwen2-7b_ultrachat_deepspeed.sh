export DATA_PATH=data/train/pure_data_tag_source_dedup_ins_tag/en/no_deduped_data/inst_ultrachat.json
export CONV_TEMPLATE=chatml-keep-system
export OUTPUT_DIR=saved_models/qwen2-7b_ultrachat
export MODEL_PATH=pretrained_models/Qwen2-7B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=4
bash scripts/train_sft.sh