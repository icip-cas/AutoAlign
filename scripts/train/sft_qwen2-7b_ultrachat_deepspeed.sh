export DATA_PATH=path/to/ultrachat.json
export CONV_TEMPLATE=chatml-keep-system
export OUTPUT_DIR=saved_models/qwen2-7b_ultrachat
export MODEL_PATH=Qwen/Qwen2-7B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=4
bash scripts/train_sft.sh
