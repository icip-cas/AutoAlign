export model_name="qwen2-7b-instruct"
export model_path="Qwen2/Qwen2-7B-Instruct"
export input="principle.json"
export output="sso_data.json"

python3 -u build_sso_data.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --input ${input} \
    --output ${output} 2>&1 | tee sso_data.log