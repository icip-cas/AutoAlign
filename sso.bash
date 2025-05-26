export WANDB_DISABLED=true

autoalign-cli sso --model_name_or_path "/run/determined/workdir/ceph_home/arknet/hf_models/Qwen/Qwen2.5-1.5B-Instruct"  \
            --data_path "data/dummy_sso.json" \
            --bf16 True \
            --output_dir "models/qwen2.5-1.5b-sso" \
            --conv_template_name chatml \
            --deepspeed "configs/zero3.json" 2>&1 | tee sso.log