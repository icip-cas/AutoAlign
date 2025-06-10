export WANDB_DISABLED=true

autoalign-cli dpo --model_name_or_path "Qwen2/Qwen2-7B-Instruct"  \
            --data_path "data/dummy_dpo.json" \
            --bf16 True \
            --output_dir "models/qwen2-7b-dpo" \
            --conv_template_name chatml \
            --deepspeed "configs/zero3.json" 2>&1 | tee sso.log