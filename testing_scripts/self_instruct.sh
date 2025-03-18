export ATA_ROOT=""

python src/autoalign/data/instruction/self_instruct.py \
    --job-id ${JOB_ID:-"llama3-8b"} \
    --template-name ${TEMPLATE_NAME:-"llama-3-instruct"} \
    --question-gen-model-path ${QGEN_MODEL:-"/ceph_home/arknet/hf_models/meta-llama/Llama-3.1-8B-Instruct/"} \
    --seed-data-path ${SEED_DATA_PATH:-"/mnt/shared_home/xudong/a800-3/auto-alignment/algorithms/self-rewarding/data/seed.json"} \
    --backend ${BACKEND:-"vllm"} \
    --num-prompts ${NUM_PROMPTS:-1}\
    --output-path ${OUTPUT_PATH:-"/141nfs/wangjunxiang/AutoAlign/testing-data/testing-output"}