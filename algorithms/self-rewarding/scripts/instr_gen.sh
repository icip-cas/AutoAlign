export ATA_ROOT="../.."

python $ATA_ROOT/src/autoalign/data/instruction/self_instruct.py \
    --job-id ${JOB_ID:-"llama3-8b"} \
    --template-name ${TEMPLATE_NAME:-"llama-3-instruct"} \
    --question-gen-model-path ${QGEN_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"} \
    --seed-data-path ${SEED_DATA_PATH:-"data/seed.json"} \
    --backend ${BACKEND:-"vllm"} \
    --num-prompts ${NUM_PROMPTS:-2}\
    --output-path ${OUTPUT_PATH:-"outputs/llama-3.1-8b-self-instruct"}
