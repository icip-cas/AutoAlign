python src/eft_en_dialogs_sample.py

python src/eft_data_sample.py \
    --model-path ${MODEL_PATH:-"saved_models/llama3-8b-ift"} \
    --model-id ${MODEL_ID:-"llama3-8b"} \
    --template-name ${TEMPLATE_NAME:-"llama-3-instruct"} \
    --seed-data-path ${SEED_DATA_PATH:-"data/seed.json"}  \
    --backend ${BACKEND:-"vllm"}
