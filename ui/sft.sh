
    autoalign-cli sft         --model_name_or_path /         --data_path /         --conv_template_name chatml         --bf16 True         --output_dir /         --num_train_epochs 3         --per_device_train_batch_size 4         --per_device_eval_batch_size 4         --gradient_accumulation_steps 1         --eval_strategy no         --eval_steps 1500         --save_strategy epoch         --learning_rate 2e-5         --weight_decay 0.01         --warmup_ratio 0.04         --lr_scheduler_type cosine         --report_to tensorboard         --logging_dir /         --logging_steps 1         --model_max_length 4096         --gradient_checkpointing True         --deepspeed Deepspeed_dir         --ddp_timeout 18000         --lazy_preprocess True         --eval_num 0         --num_workers 1 |& tee cai_sft.log
    