export WANDB_DISABLED=true

autoalign-cli sso --model_name_or_path "Qwen2/Qwen2-7B-Instruct"  \
        --data_path "sso_data.json" \
        --bf16 True \
        --output_dir "models/Qwen2-7B-Generator" \
        --conv_template_name chatml \
        --deepspeed "../../configs/zero3.json" \
        --num_train_epochs "1" \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 5e-7 \
        --max_prompt_length 1024 \
        --max_length 2048 \
        --weight_decay 0.1 \
        --warmup_ratio 0.04 \
        --lr_scheduler_type "cosine" \
        --report_to "tensorboard" \
        --logging_dir "models/Qwen2-7B-Generator" \
        --logging_steps 1 \
        --gradient_checkpointing True  2>&1 | tee sso.log