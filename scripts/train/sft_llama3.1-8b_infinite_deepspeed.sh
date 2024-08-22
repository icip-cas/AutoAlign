export DATA_PATH=data/train/infinite_7m.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=./saved_models/llama-31-8b_ultrachat_infinite
export MODEL_PATH=pretrained_models/Meta-Llama-3.1-8B
export GA=16
export DS_CONFIG=configs/zero2.json
export LR=5e-6
export TRAIN_BATCH_SIZE=4
export WARMUP_RATIO=1e-5
export EPOCH=1
export LAZY="True"
bash scripts/train_sft.sh
