export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=mistral-instruct
export OUTPUT_DIR=/ciphome/wenxueru2022/auto-alignment/checkpoints/Mistral-7B-v0.3
export MODEL_PATH=/ciphome/wenxueru2022/auto-alignment/hf_models/Mistral-7B-v0.3
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh