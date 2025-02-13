autoalign-cli sft \
            --model_name_or_path "Qwen2/Qwen2-7B" \
            --data_path "data/dummy_sft.json" \
            --bf16 True \
            --output_dir "models/qwen2-7b-sft" \
            --model_max_length 4096 \
            --conv_template_name chatml \
            --deepspeed "configs/zero3.json"

autoalign-cli rm --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
            --data_path data/ultra_binary.jsonl \
            --bf16 True \
            --eval_path data/eval \
            --conv_template_name llama-3-instruct \
            --output_dir models/llama3_rm \
            --deepspeed configs/zero3.json

autoalign-cli dpo --model_name_or_path "Qwen2/Qwen2-7B-Instruct"  \
            --data_path "data/dummy_dpo.json" \
            --bf16 True \
            --output_dir "models/qwen2-7b-dpo" \
            --conv_template_name chatml \
            --deepspeed "configs/zero3.json"

autoalign-cli infer --backend "vllm" \
            --model-name "Qwen2-0.5B-Instruct" \
            --model-path "Qwen/Qwen2-0.5B-Instruct" \
            --test-file "data/dummy_sft.json" \
            --template "chatml" \
            --source "qwen2_0_5b_instruct_dummy"

autoalign-cli serve --checkpoint-path "Qwen2/Qwen2-7B-Instruct" \
            --mode "browser" \
            --template "chatml" \

autoalign-cli merge --model_paths "psmathur/orca_mini_v3_13b" "WizardLM/WizardLM-13B-V1.2" "garage-bAInd/Platypus2-13B" \
                    --merged_model_path "merged_model" \
                    --merging_method "average"