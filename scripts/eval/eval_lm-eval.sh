#!/bin/bash

export NCCL_CUMEM_ENABLE=0
export HF_ENDPOINT="https://hf-mirror.com"
export HF_DATASETS_CACHE="./caches/hf_cache/datasets"

declare -A MODELS

MODELS["mistral-01-7b_ultrachat"]="saved_models/mistral-01-7b_ultrachat"
# Place more models here

TASKS=("arc_challenge" "hellaswag" "truthfulqa" "gsm8k" "mmlu" "winogrande")
FEWSHOTS=("25" "10" "0" "5" "5" "5")

for MODEL_NAME in "${!MODELS[@]}"; do
  MODEL_PATH="${MODELS[$MODEL_NAME]}"
  for i in "${!TASKS[@]}"; do
    TASK="${TASKS[$i]}"
    NUM_FEWSHOT="${FEWSHOTS[$i]}"
    OUTPUT_PATH="./outputs/${MODEL_NAME}/${TASK}.json"

    echo "Evaluating model: ${MODEL_NAME} at path: ${MODEL_PATH}"

    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${MODEL_PATH}" \
        --tasks "${TASK}" \
        --batch_size auto \
        --num_fewshot "${NUM_FEWSHOT}" \
        --output_path "${OUTPUT_PATH}" \
        --trust_remote_code

  done
done
