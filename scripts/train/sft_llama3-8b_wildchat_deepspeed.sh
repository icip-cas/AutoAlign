export DATA_PATH=path/to/wildchat.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=saved_models/llama-3-8b_wildchat
export MODEL_PATH=meta-llama/Meta-Llama-3-8B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=4
bash scripts/train_sft.sh
