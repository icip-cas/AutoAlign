export DATA_PATH=/ciphome/wenxueru2022/auto-alignment/data/ultrachat.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=/ciphome/wenxueru2022/auto-alignment/checkpoints/Meta-Llama-3-8B
export MODEL_PATH=/data7/hf_models/NousResearch/Meta-Llama-3-8B
export GA=8
export DS_CONFIG=configs/zero2.json
export TRAIN_BATCH_SIZE=2
bash scripts/train_sft.sh