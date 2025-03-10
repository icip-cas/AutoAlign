
python src/autoalign/data/instruction/self_instruct.py \
    --model-id tests \
    --template-name qwen \
    --question-gen-model-path /141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --seed-data-path /141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --backend hf \
    --num-prompts 10\
    --output-path /141nfs/maoyingzhi2024/hf_models/Qwen/Qwn2.5-1.5B-instruct/

