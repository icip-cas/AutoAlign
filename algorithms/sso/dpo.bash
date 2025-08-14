export WANDB_DISABLED=true

autoalign-cli dpo --model_name_or_path "Qwen2/Qwen2-7B-Instruct" \
    --data_path "dpo_data.json" \
    --conv_template_name chatml \
    --bf16 True \
    --output_dir "models/Qwen2-7B-SSO" \
    --num_train_epochs "1" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate 5e-7 \
    --max_prompt_length 1024 \
    --max_length 2048 \
    --weight_decay 0.1 \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --logging_dir "models/Qwen2-7B-SSO" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed "../../configs/zero3.json" 2>&1 | tee dpo.log