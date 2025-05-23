import streamlit as st
import os
import time
from pages.navigation import render_navbar, init_session_state


render_navbar()
hide_sidebar_css = """
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_sidebar_css, unsafe_allow_html=True)
if "sub_sel_button" not in st.session_state:
    st.session_state.sub_sel_button = "SFT"
st.session_state.sub_sel_button = st.selectbox(
    "Configure: ",
    ("SFT", "DPO", "CAI_sft", "CAI_dpo", "SPIN", "RLCD", "Self_Rewarding"),
)
with st.form("Train Configuration"):
    if st.session_state.sub_sel_button in ["SFT", "CAI_sft"]:
        cols = st.columns([3, 1, 1, 1])
        with cols[0]:
            proj_name = st.text_input("Project Name", value="aaa")
            model = st.text_input("Base Model", value="/141nfs/arknet/hf_models/Qwen1.5-1.8B-Chat/")
            dataset = st.text_input("Dataset Source", value="ata-output")
            output_dir = st.text_input("Output Dir", value="ata-output")
            st.session_state.model_dir = output_dir
        with cols[1]:
            train_batch_size = st.number_input("Train Batch Size per Device", value=4)
            chat_template = st.selectbox(
                "Template Select",
                (
                    "chatml",
                    "vicuna_v1.1",
                    "llama-2-chat",
                    "llama-2-chat-keep-system",
                    "chatml-keep-system",
                    "llama-3-instruct",
                    "mistral-instruct",
                    "gemma",
                    "zephyr",
                    "chatml-idsys",
                    "glm-4-chat",
                    "glm-4-chat-keep-system",
                    "default",
                ),
            )
            eval_strategy = st.selectbox("Eval Strategy", ("no", "steps"))
            bf16 = st.selectbox("bf16", ("True", "False"))
        with cols[2]:
            learning_rate = st.text_input("Learning Rate", value="2e-5")
            lr_scheduler = st.selectbox("LR Scheduler", ("cosine"))
            logging_steps = st.number_input("Logging Steps", value=1)
            deepspeed = st.text_input("Deepspeed", value="configs/zero3.json")
        with cols[3]:
            model_max_length = st.number_input("Model Max Length", value=4096)
            eval_num = st.number_input("Eval Num", value=0)
            num_train_epoch = st.number_input("Train Epoch", value=3)
            eval_batch_size = st.number_input("Eval Batch Size", value=4)
        cols = st.columns(6)
        with cols[0]:
            eval_steps = st.number_input("Eval Steps", value=1500)
            weight_decay = st.number_input("Weight Decay", value=0.01)
        with cols[1]:
            report_to = st.selectbox("Report To", ("tensorboard"))
            ddp_timeout = st.number_input("DDP Timeout", value=18000)
        with cols[2]:
            num_workers = st.number_input("Worker Num", value=1)
            gradient_accumulation_steps = st.number_input("GA", value=1)
        with cols[3]:
            save_strategy = st.selectbox("Save Strategy", ("epoch"))
            warmup_ratio = st.number_input("Warmup Ratio", value=0.04)
        with cols[4]:
            logging_dir = st.text_input("Logging Dir", value="ata-output")
            gradient_checkpoint = st.selectbox("Gradient Checkpoint", (True, False))
        with cols[5]:
            lazy_preprocess = st.selectbox("Lazy Preprocess", (True, False))

    elif st.session_state.sub_sel_button in [
        "DPO",
        "CAI_dpo",
        "RLCD",
        "SPIN",
        "Self_Rewarding",
    ]:
        proj_name = st.text_input("Project Name", value="aaa")
        model = st.text_input("Base Model", value="/141nfs/arknet/hf_models/Qwen1.5-1.8B-Chat/")
        dataset = st.text_input("Dataset Source", value="ata-output")  # 对应 --data_path
        output_dir = st.text_input("Output Dir", value="ata-output")
        train_batch_size = st.number_input("Train Batch Size per Device", value=4)
        # 添加 beta 参数
        beta = st.number_input("Beta", value=0.1, step=0.01, format="%.2f")
        chat_template = st.selectbox(
            "Template Select",
            (
                "chatml",
                "vicuna_v1.1",
                "llama-2-chat",
                "llama-2-chat-keep-system",
                "chatml-keep-system",
                "llama-3-instruct",
                "mistral-instruct",
                "gemma",
                "zephyr",
                "chatml-idsys",
                "glm-4-chat",
                "glm-4-chat-keep-system",
                "default",
            ),
        )
        eval_strategy = st.selectbox("Eval Strategy", ("no"))
        bf16 = st.selectbox("bf16", ("True", "False"))
        lr_scheduler = st.selectbox("LR Scheduler", ("cosine"))
        logging_steps = st.number_input("Logging Steps", value=1)
        deepspeed = st.text_input(
            "Deepspeed", value="configs/zero3.json"
        )  # 更明确的默认值

        learning_rate = st.text_input("Learning Rate", value="5e-7")  # 改为 DPO 典型值
        num_train_epoch = st.number_input("Train Epoch", value=1)
        eval_batch_size = st.number_input("Eval Batch Size", value=4)
        eval_steps = st.number_input("Eval Steps", value=1500)
        weight_decay = st.number_input("Weight Decay", value=0.01)
        report_to = st.selectbox(
            "Report To",
            ("tensorboard", "wandb"),  # 扩展选项
        )
        # 添加模型保存相关参数
        save_steps = st.number_input("Save Steps", value=400)
        save_total_limit = st.number_input("Save Total Limit", value=100)

        gradient_accumulation_steps = st.number_input("GA", value=1)
        save_strategy = st.selectbox(
            "Save Strategy",
            ("epoch", "steps"),  # 增加 steps 选项
        )
        warmup_ratio = st.number_input(
            "Warmup Ratio", value=0.04, min_value=0.0
        )  # 修正负数问题
        logging_dir = st.text_input("Logging Dir", value="ata-output")
        gradient_checkpoint = st.selectbox("Gradient Checkpoint", (True, False))

    if st.form_submit_button("→"):
        if st.session_state.sub_sel_button in [
            "DPO",
            "CAI_dpo",
            "RLCD",
            "SPIN",
            "Self_Rewarding",
        ]:
            missing_fields = []
            # 检查文本字段是否为空
            if not proj_name.strip():
                missing_fields.append("Project Name")
            if not model.strip():
                missing_fields.append("Base Model")
            if not dataset.strip():
                missing_fields.append("Dataset Source")
            if not output_dir.strip():
                missing_fields.append("Output Dir")

            # 数值型参数检查
            if train_batch_size <= 0:
                missing_fields.append("Train Batch Size per Device")
            if beta <= 0:  # DPO特有的beta参数需为正
                missing_fields.append("Beta")
            if logging_steps <= 0:
                missing_fields.append("Logging Steps")
            if num_train_epoch <= 0:
                missing_fields.append("Train Epoch")
            if eval_batch_size <= 0:
                missing_fields.append("Eval Batch Size")
            if eval_steps <= 0:
                missing_fields.append("Eval Steps")
            if weight_decay <= 0:  # 同SFT逻辑，需大于0
                missing_fields.append("Weight Decay")
            if save_steps <= 0:
                missing_fields.append("Save Steps")
            if save_total_limit < 0:  # 允许0表示不限制但不可为负
                missing_fields.append("Save Total Limit")
            if gradient_accumulation_steps <= 0:
                missing_fields.append("GA")
            if warmup_ratio <= 0:  # 同SFT逻辑，需正数
                missing_fields.append("Warmup Ratio")

            # 路径/配置检查
            if not deepspeed.strip():
                missing_fields.append("Deepspeed")
            if not logging_dir.strip():
                missing_fields.append("Logging Dir")

            # 学习率格式验证
            if not learning_rate.strip():
                missing_fields.append("Learning Rate")
            else:
                try:
                    lr_val = float(learning_rate)
                    if lr_val <= 0:
                        missing_fields.append("Learning Rate (must be positive)")
                except ValueError:
                    missing_fields.append("Learning Rate (invalid format)")

            # 选项类参数检查
            if gradient_checkpoint is None:  # 确保已选择梯度检查点
                missing_fields.append("Gradient Checkpoint")

            # 检查路径合法性
            invalid_paths = []
            if model and not os.path.exists(model):
                invalid_paths.append(f"Model Path '{model}' does not exist.")
            if dataset and not os.path.exists(dataset):
                invalid_paths.append(f"Dataset Path '{dataset}' does not exist.")

            # 如果有缺失字段或无效路径，显示详细的错误信息
            if missing_fields or invalid_paths:
                error_message = "Please fix the following errors before saving:\n\n"

                if missing_fields:
                    error_message += "**Missing Fields:**\n"
                    for field in missing_fields:
                        error_message += f"- {field} is required.\n"

                if invalid_paths:
                    error_message += "\n**Invalid Paths:**\n"
                    for path_error in invalid_paths:
                        error_message += f"- {path_error}\n"

                st.error(error_message)

            else:
                script_content = f"""
autoalign-cli dpo \
    --model_name_or_path {model} \
    --data_path {dataset} \
    --conv_template_name {chat_template} \
    --bf16 {bf16} \
    --output_dir {output_dir} \
    --num_train_epochs {num_train_epoch} \
    --per_device_train_batch_size {train_batch_size} \
    --per_device_eval_batch_size {eval_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_strategy {eval_strategy} \
    --eval_steps {eval_steps} \
    --save_strategy {save_strategy} \
    --save_steps {save_steps} \
    --save_total_limit {save_total_limit} \
    --learning_rate {learning_rate} \
    --beta {beta} \
    --weight_decay {weight_decay} \
    --warmup_ratio {warmup_ratio} \
    --lr_scheduler_type {lr_scheduler} \
    --report_to {report_to} \
    --logging_dir {logging_dir} \
    --logging_steps {logging_steps} \
    --gradient_checkpointing {gradient_checkpoint} \
    --deepspeed {deepspeed} 2>&1 | tee outputs/cai_sft.log; echo "###page8###" >> outputs/cai_sft.log
"""

                # current_dir = os.path.dirname(os.path.abspath(__file__))
                # # current_dir = "/141nfs/wangpengbo/auto_alignment/auto-alignment/algorithms/cai"
                # if st.session_state.sub_sel_button == 'CAI_dpo':
                #     bash_file_path = os.path.join(current_dir, "cai_dpo.sh")
                # elif st.session_state.sub_sel_button == 'DPO':
                #     bash_file_path = os.path.join(current_dir, "dpo.sh")
                # elif st.session_state.sub_sel_button == 'RLCD':
                #     bash_file_path = os.path.join(current_dir, "rlcd.sh")
                # elif st.session_state.sub_sel_button == 'SPIN':
                #     bash_file_path = os.path.join(current_dir, "spin.sh")
                # elif st.session_state.sub_sel_button == 'Self_Rewarding':
                #     bash_file_path = os.path.join(current_dir, "self_rewarding.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                st.session_state.step3 = script_content
                st.session_state.p3_fin = True
                st.success("Configuration Saved!")
                st.session_state.selected_button = "eval"
                time.sleep(1.5)
                st.switch_page("pages/evaluation_ui.py")

        if st.session_state.sub_sel_button in ["SFT", "CAI_sft"]:
            # Temporary skipping part
            missing_fields = []
            if not proj_name.strip():  #
                missing_fields.append("Project Name")
            if not model.strip():  #
                missing_fields.append("Base Model")
            if not dataset.strip():  #
                missing_fields.append("Dataset Source")
            if not output_dir.strip():  #
                missing_fields.append("Output Dir")
            if train_batch_size <= 0:  #
                missing_fields.append("Train Batch Size per Device")
            if not lr_scheduler:  #
                missing_fields.append("LR Scheduler")
            if logging_steps <= 0:  #
                missing_fields.append("Logging Steps")
            if not deepspeed.strip():  #
                missing_fields.append("Deepspeed")
            if eval_num < 0:  #
                missing_fields.append("Eval Num")
            if not learning_rate.strip():  #
                missing_fields.append("Learning Rate")
            if num_train_epoch <= 0:  #
                missing_fields.append("Train Epoch")
            if eval_batch_size <= 0:  #
                missing_fields.append("Eval Batch Size")
            if eval_steps <= 0:  #
                missing_fields.append("Eval Steps")
            if weight_decay <= 0:  #
                missing_fields.append("Weight Decay")
            if not report_to:  #
                missing_fields.append("Report To")
            if ddp_timeout <= 0:  #
                missing_fields.append("DDP Timeout")
            if num_workers <= 0:  #
                missing_fields.append("Worker Num")
            if model_max_length <= 0:  #
                missing_fields.append("Model Max Length")
            if gradient_accumulation_steps <= 0:  #
                missing_fields.append("GA")
            if not save_strategy:  #
                missing_fields.append("Save Strategy")
            if warmup_ratio <= 0:  #
                missing_fields.append("Warmup Ratio")
            if not logging_dir.strip():  #
                missing_fields.append("Logging Dir")
            if gradient_checkpoint is None:  #
                missing_fields.append("Gradient Checkpoint")
            if lazy_preprocess is None:  #
                missing_fields.append("Lazy Preprocess")


            # 检查路径合法性
            invalid_paths = []
            if model and not os.path.exists(model):
                invalid_paths.append(f"Model Path '{model}' does not exist.")
            if dataset and not os.path.exists(dataset):
                invalid_paths.append(f"Dataset Path '{dataset}' does not exist.")

            # 如果有缺失字段或无效路径，显示详细的错误信息
            if missing_fields or invalid_paths:
                error_message = "Please fix the following errors before saving:\n\n"

                if missing_fields:
                    error_message += "**Missing Fields:**\n"
                    for field in missing_fields:
                        error_message += f"- {field} is required.\n"

                if invalid_paths:
                    error_message += "\n**Invalid Paths:**\n"
                    for path_error in invalid_paths:
                        error_message += f"- {path_error}\n"

                st.error(error_message)

            else:
                script_content = f"""
autoalign-cli sft \
    --model_name_or_path {model} \
    --data_path {dataset} \
    --conv_template_name {chat_template} \
    --bf16 {bf16} \
    --output_dir {output_dir} \
    --num_train_epochs {num_train_epoch} \
    --per_device_train_batch_size {train_batch_size} \
    --per_device_eval_batch_size {eval_batch_size} \
    --gradient_accumulation_steps {gradient_accumulation_steps} \
    --eval_strategy {eval_strategy} \
    --eval_steps {eval_steps} \
    --save_strategy {save_strategy} \
    --learning_rate {learning_rate} \
    --weight_decay {weight_decay} \
    --warmup_ratio {warmup_ratio} \
    --lr_scheduler_type {lr_scheduler} \
    --report_to {report_to} \
    --logging_dir {logging_dir} \
    --logging_steps {logging_steps} \
    --model_max_length {model_max_length} \
    --gradient_checkpointing {gradient_checkpoint} \
    --deepspeed {deepspeed} \
    --ddp_timeout {ddp_timeout} \
    --lazy_preprocess {lazy_preprocess} \
    --eval_num {eval_num} \
    --num_workers {num_workers} 2>&1 | tee outputs/cai_sft.log; echo "###page8###" >> outputs/cai_sft.log
"""
                st.session_state.step3 = script_content
                st.session_state.p3_fin = True
                st.success("Configuration Saved!")
                st.session_state.selected_button = "eval"
                time.sleep(1.5)
                st.switch_page("pages/evaluation_ui.py")


if st.session_state.selected_button == "data_gen":
    st.switch_page("pages/instruction_generation_ui.py")
elif st.session_state.selected_button == "data_filter":
    st.switch_page("pages/sampling_answer_ui.py")
elif st.session_state.selected_button == "train":
    pass
elif st.session_state.selected_button == "eval":
    if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.switch_page("pages/evaluation_ui.py")
