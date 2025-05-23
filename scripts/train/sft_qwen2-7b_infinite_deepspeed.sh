export DATA_PATH=path/to/infinite_7m.json
export CONV_TEMPLATE=chatml-keep-system
export OUTPUT_DIR=saved_models/qwen2-7b_infinite
export MODEL_PATH=Qwen/Qwen2-7B
export GA=16
export DS_CONFIG=configs/zero2.json
export LR=5e-6
export TRAIN_BATCH_SIZE=4
export REPORT_TO="wandb"
export WARMUP_RATIO=1e-5
export EPOCH=1
export LAZY="True"
bash scripts/train_sft.sh
