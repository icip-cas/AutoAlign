export DATA_PATH=path/to/ultrachat.json
export CONV_TEMPLATE=glm-4-chat-keep-system
export OUTPUT_DIR=saved_models/glm-4-9b_ultrachat
export MODEL_PATH=THUDM/glm-4-9b
export GA=16
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh
