
#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0

<<<<<<< Updated upstream
model_path=/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/
data_filepath=/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/
save_filepath=/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/
prompt_column_name=141nfs/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/

python src/autoalign/data/instruction/back_translation.py \
    --reverse \
    --model_path=/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --data_filepath=/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --save_filepath=/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
    --prompt_column_name=141nfs/141nfs/maoyingzhi2024/hf_models/Qwen/Qwen2.5-1.5B-instruct/ \
=======
model_path=/mnt/ceph_home/arknet/hf_models/Qwen/Qwen2.5-1.5B-Instruct
data_filepath=/mnt/ceph_home/xudong2022/auto-alignment/data/Humback/falcon-refinedweb-sampled.jsonl
save_filepath=outputs/unlabelled_gen_instruction.jsonl
prompt_column_name=content
tensor_parallel_size=1

python src/autoalign/data/instruction/back_translation.py \
    --reverse \
    --model_path=/mnt/ceph_home/arknet/hf_models/Qwen/Qwen2.5-1.5B-Instruct \
    --data_filepath=/mnt/ceph_home/xudong2022/auto-alignment/data/Humback/falcon-refinedweb-sampled.jsonl \
    --save_filepath=outputs/unlabelled_gen_instruction.jsonl \
    --prompt_column_name=content \
>>>>>>> Stashed changes
    --tensor_parallel_size=1
