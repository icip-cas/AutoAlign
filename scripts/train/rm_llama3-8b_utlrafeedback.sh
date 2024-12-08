export DATA_PATH=data/ultra_binary.jsonl
export EVAL_DATA_PATH=data/eval
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=saved_models/llama-3-8b_ultrafeedback_rm/
export MODEL_PATH=pretrained_models/Meta-Llama-3-8B
export GA=4
export TRAIN_BATCH_SIZE=2
bash scripts/train_rm.sh
