
python src/autoalign/data/instruction/self_instruct.py \
<<<<<<< Updated upstream
    --model-id tests \
    --template-name qwen \
    --question-gen-model-path /141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --seed-data-path /141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --backend hf \
    --num-prompts 10\
    --output-path /141nfs/maoyingzhi2024/hf_models/Qwen/Qwn2.5-1.5B-instruct/
=======
    --model-id Qwen2 \
    --template-name Qwen \
    --question-gen-model-path /mnt/ceph_home/arknet/hf_models/Qwen/Qwen2.5-1.5B-Instruct \
    --seed-data-path /mnt/shared_home/xudong/a800-3/auto-alignment/algorithms/self-rewarding/data/seed.json \
    --backend vllm \
    --num-prompts 10\
    --output-path outputs/qwen-1.5b
>>>>>>> Stashed changes

