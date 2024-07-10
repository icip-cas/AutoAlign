export modelname=$1
export modelpath=$2

export batch_size=8
export HF_ENDPOINT=https://hf-mirror.com
export overwrite=True
export OPENAI_MAX_CONCURRENCY=8
unset VLLM_USE_MODELSCOPE

if [ ! -d "eval_result/alpaca/${modelname}" ]; then
    mkdir -p eval_result/alpaca/${modelname}
fi

if [ ! -f "eval_result/alpaca/${modelname}/${modelname}_outputs.json" ] || [ $overwrite = "True" ]; then
    echo "Generating model outputs..."
    python eval_utils/generate_output.py \
        --model $modelpath \
        --batch_size $batch_size \
        --output_path eval_result/alpaca/${modelname}/${modelname}_outputs.json \
        --name $modelname
fi

alpaca_eval --model_outputs eval_result/alpaca/${modelname}/${modelname}_outputs.json \
    --output_path eval_result/alpaca/${modelname}/ \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --is_overwrite_leaderboard