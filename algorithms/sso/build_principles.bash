export model_name="qwen2-7b-instruct"
export model_path="Qwen2/Qwen2-7B-Instruct"
export input="dummy_sso_prompt.json"
export output="principle.json"

python build_principles.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --input ${input} \
    --output ${output} 2>&1 | tee principle.log