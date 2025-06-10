export model_name=
export model_path=
export input=
export output=

python3 -u build_sso_data.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --input ${input} \
    --output ${output}