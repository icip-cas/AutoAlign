export DATA_PATH="data/dummy_sft.json"
# export CONV_TEMPLATE=chatml-keep-system
export OUTPUT_DIR="output/model/qwen2.5-7b"
export MODEL_PATH="model/Qwen2.5-7B"
export GA=32
export DS_CONFIG="configs/zero3.json"
export TRAIN_BATCH_SIZE=2
export logging_step=1
export MAX_LENGTH=4096
bash scripts/train_sft.sh

