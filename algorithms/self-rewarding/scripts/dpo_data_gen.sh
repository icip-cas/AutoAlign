python src/dpo_dataset_generator.py \
    --model-path ${MODEL_PATH:-"saved_models/llama3-8b-eft"} \
    --model-id ${MODEL_ID:-"llama3-8b"} \
    --template-name ${TEMPLATE_NAME:-"llama-3-instruct"} \
    --sft-base-model ${SFT_BASE_MODEL:-"eft"} \
    --backend ${BACKEND:-"vllm"} \
    --num-iter ${NUM_ITER:-1} \
    --instruction-path ${INS_PATH:-"data/seed.json"}
