export DATA_PATH=path/to/ultrachat.json
export CONV_TEMPLATE=llama-2-chat-keep-system
export OUTPUT_DIR=saved_models/llama-2-7b_ultrachat/
export MODEL_PATH=meta-llama/Llama-2-7b-hf/
export GA=4
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=8
bash scripts/train_sft.sh
