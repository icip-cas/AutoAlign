python src/dpo_dataset_generator.py \
    --model-path ${MODEL_PATH:-"saved_models/llama3-8b-eft"} \
    --model-id ${MODEL_ID:-"llama3-8b"} \
    --template-name ${TEMPLATE_NAME:-"llama-3-instruct"} \
    --question-gen-model-path ${QGEN_MODEL:-"/mnt/userdata/hf_models/NousResearch/Meta-Llama-3-8B-Instruct"} \
    --seed-data-path ${SEED_DATA_PATH:-"data/seed.json"} \
    --sft-base-model ${SFT_BASE_MODEL:-"eft"} \
    --backend ${BACKEND:-"vllm"} \
    --num-iter ${NUM_ITER:-2} \
    --num-prompts ${NUM_PROMPTS:-2000}\
