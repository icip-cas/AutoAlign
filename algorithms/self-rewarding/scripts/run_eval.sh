python src/eval.py \
    --model-path ${MODEL_PATH:-"path/to/model"} \
    --model-name ${MODEL_NAME:-"name-of-model"} \
    --template-name ${TEMPLATE_NAME:-"llama-3-instruct"} \
    --opencompass-path ${OPENCOMPASS_PATH:-"../../opencompass"} \
    --num-gpus-per-model ${NUM_GPUS_PER_MODEL:-1} \
    --batch-size ${BATCH_SIZE:-8}\
    --eval-type ${EVAL_TYPE:-"objective"} \
    --eval-judge-data ${EVAL_JUDGE_DATA:-"path/to/eval_eft_data.json"} \
    # --subjective_generate_only \
    # --reuse ${REUSE:-"path/to/reuse/folder"} \
