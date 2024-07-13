export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=qwen-7b-chat-keep-system
export OUTPUT_DIR=/ciphome/wenxueru2022/auto-alignment/checkpoints/Qwen2-7B
export MODEL_PATH=/data7/hf_models/qwen/Qwen2-7B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh