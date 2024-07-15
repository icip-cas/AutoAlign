export DATA_PATH=./data/train/pure_data_tag_source_dedup_ins_tag/en/no_deduped_data/inst_ultrachat.json
export CONV_TEMPLATE=mistral-instruct
export OUTPUT_DIR=./saved_models/
export MODEL_PATH=/ciphome/wenxueru2022/auto-alignment/hf_models/Mistral-7B-v0.3
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh