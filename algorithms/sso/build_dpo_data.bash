export model_name="qwen2-7b-generator"
export model_path="models/Qwen2-7B-Generator"
export input="principle.json"
export output="dpo_data.json"

python3 -u build_dpo_data.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --input ${input} \
    --output ${output} 2>&1 | tee dpo_data.log 