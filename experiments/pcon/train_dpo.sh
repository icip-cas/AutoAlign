autoalign-cli dpo --model_name_or_path "./pretrained_models/Qwen2-7B"  \
    --data_path "./outputs/qwen2-7b_pcon_ultrafeedback_ata.json" \
    --conv_template_name "chatml" \
    --bf16 True \
    --output_dir "./saved_models/qwen2-7b_pcon_ultrafeedback" \
    --num_train_epochs "1" \
    --per_device_train_batch_size "1" \
    --per_device_eval_batch_size "4" \
    --gradient_accumulation_steps "4" \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate "2e-5" \
    --max_prompt_length 1024 \
    --max_length 2048 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --logging_dir "./saved_models/qwen2-7b_pcon_ultrafeedback" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed "../../configs/zero3.json"