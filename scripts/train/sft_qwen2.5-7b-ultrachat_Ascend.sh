export DATA_PATH="data/dummy_sft.json"
# export CONV_TEMPLATE=chatml-keep-system
export OUTPUT_DIR="/mnt/data3/maoyingzhi2024/ATA-Ascend/AutoAlign/model/"
export MODEL_PATH="/mnt/data1/hf_models/Qwen2.5-1.5B"
export GA=32
export DS_CONFIG="configs/zero3.json"
export TRAIN_BATCH_SIZE=1
export logging_step=1
export MAX_LENGTH=1024
bash scripts/train_sft.sh

