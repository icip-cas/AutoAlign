export HF_DATASETS_CACHE="./caches/hf_cache/datasets"
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="gemma-2-9b"
MODEL_PATH="google/gemma-2-9b"

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "leaderboard_bbh" \
    --batch_size auto \
    --output_path "./outputs/${MODEL_NAME}/leaderboard_bbh" \
    --trust_remote_code \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "leaderboard_gpqa" \
    --batch_size auto \
    --output_path "./outputs/${MODEL_NAME}/leaderboard_gpqa" \
    --trust_remote_code \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "leaderboard_ifeval" \
    --batch_size auto \
    --output_path "./outputs/${MODEL_NAME}/leaderboard_ifeval" \
    --trust_remote_code \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "leaderboard_math_hard" \
    --batch_size auto \
    --output_path "./outputs/${MODEL_NAME}/leaderboard_math_hard" \
    --trust_remote_code \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "leaderboard_mmlu_pro" \
    --batch_size auto \
    --output_path "./outputs/${MODEL_NAME}/leaderboard_mmlu_pro" \
    --trust_remote_code \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "leaderboard_musr" \
    --batch_size auto \
    --output_path "./outputs/${MODEL_NAME}/leaderboard_musr" \
    --trust_remote_code \
