#!/bin/bash
export WANDB_DISABLED=true

export principle_model_name="qwen2-7b-instruct"
export principle_model_path="Qwen2/Qwen2-7B-Instruct"
export model_name="qwen2-7b-instruct"
export model_path="Qwen2/Qwen2-7B-Instruct"
export model_generator_path="models/Qwen2-7B-Generator"
export principle_input="prompt.json"
export principle_output=sso_data_input=dpo_data_input="principle.json"
export sso_data_output=sso_train_input="sso_data.json"
export dpo_data_output=dpo_train_input="dpo_data.json"
export model_output_path="models/Qwen2-7B-SSO"

python build_principles.py \
    --model_name ${principle_model_name} \
    --model_path ${principle_model_path} \
    --input ${principle_input} \
    --output ${principle_output} 2>&1 | tee principle.log

python3 -u build_sso_data.py \
    --model_name ${model_name} \
    --model_path ${model_path} \
    --input ${sso_data_input} \
    --output ${sso_data_output} 2>&1 | tee sso_data.log

autoalign-cli sso --model_name_or_path ${model_path} \
        --data_path ${sso_train_input} \
        --bf16 True \
        --output_dir ${model_generator_path} \
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
        --logging_dir ${model_generator_path} \
        --logging_steps 1 \
        --gradient_checkpointing True  2>&1 | tee sso.log

python3 -u build_dpo_data.py \
    --model_name ${model_generator_path} \
    --model_path ${model_generator_path} \
    --input ${dpo_data_input} \
    --output ${dpo_data_output} 2>&1 | tee dpo_data.log

autoalign-cli dpo --model_name_or_path ${model_path} \
    --data_path ${dpo_train_input} \
    --conv_template_name chatml \
    --bf16 True \
    --output_dir ${model_output_path} \
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
    --logging_dir ${model_output_path} \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed "../../configs/zero3.json" 2>&1 | tee dpo.log