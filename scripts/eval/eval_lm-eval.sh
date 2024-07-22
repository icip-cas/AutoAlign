export NCCL_CUMEM_ENABLE=0
export HF_ENDPOINT="https://hf-mirror.com"

export HF_DATASETS_CACHE="./caches/hf_cache/datasets"

MODEL_NAME="qwen2-7b_ultrachat"
MODEL_PATH="saved_models/qwen2-7b_ultrachat"

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "arc_challenge" \
    --batch_size auto \
    --num_fewshot 25 \
    --output_path "./outputs/${MODEL_NAME}/arc_challenge.json" \
    --trust_remote_code

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "hellaswag" \
    --batch_size auto \
    --num_fewshot 10 \
    --output_path "./outputs/${MODEL_NAME}/hellaswag.json" \
    --trust_remote_code

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "truthfulqa" \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "./outputs/${MODEL_NAME}/truthfulqa_mc.json" \
    --trust_remote_code

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "gsm8k" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/gsm8k.json" \
    --trust_remote_code

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "mmlu" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/mmlu.json" \
    --trust_remote_code

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "winogrande" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/winogrande.json" \
    --trust_remote_code