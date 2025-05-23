export DATA_PATH=path/to/infinite_7m.json
export CONV_TEMPLATE=llama-2-chat-keep-system
export OUTPUT_DIR=saved_models/tinyllama_infinite
export MODEL_PATH=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
export GA=4
export DS_CONFIG=configs/zero2.json
export REPORT_TO="wandb"
export LR=5e-6
export TRAIN_BATCH_SIZE=16
export WARMUP_RATIO=1e-5
export EPOCH=1
export LAZY="True"
bash scripts/train_sft.sh
