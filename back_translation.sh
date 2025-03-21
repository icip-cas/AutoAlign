
#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES=0

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
    --tensor_parallel_size=1
