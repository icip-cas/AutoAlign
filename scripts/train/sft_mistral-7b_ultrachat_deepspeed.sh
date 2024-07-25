export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=mistral-instruct
export OUTPUT_DIR=./saved_models/mistral-03-7b_ultrachat
export MODEL_PATH=/ciphome/wenxueru2022/auto-alignment/hf_models/Mistral-7B-v0.3
export GA=2
export LR=3e-6
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=8
bash scripts/train_sft.sh