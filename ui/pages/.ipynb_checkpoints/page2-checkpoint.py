import streamlit as st
import os
import time
from pages.navbar import render_navbar, check_and_switch_page_2, init_session_state


render_navbar()
hide_sidebar_css = """
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_sidebar_css, unsafe_allow_html=True)
st.session_state.method = st.selectbox(
    "Sample Method", ("RLCD", "SPIN", "CAI_sft", "CAI_dpo", "Self_Rewarding")
)
if st.session_state.method == "RLCD":
    with st.form("input"):
        st.write("prepare for rlcd")
        cols = st.columns(3)
        with cols[0]:
            st.session_state.rlcd_input_file_dir = st.text_input("Prompt File", value="testing-output")
        with cols[1]:
            st.session_state.rlcd_output_chosen = st.text_input("Output Chosen", value="testing-output")
        with cols[2]:
            st.session_state.rlcd_output_rejected = st.text_input("Output Rejected", value="testing-output")
        st.write("inference for rlcd")
        cols = st.columns(4)
        with cols[0]:
            st.session_state.rlcd_infer_model_name = st.text_input("Model name", value="Qwen2.5-Ins")
            st.session_state.rlcd_infer_model_path = st.text_input("Model path", value="/141nfs/arknet/hf_models/Qwen1.5-1.8B-Chat/")
        with cols[1]:
            st.session_state.rlcd_infer_chosen = st.text_input("Output chosen file", value="testing-output")
            st.session_state.rlcd_infer_rejected = st.text_input("Output rejected file", value="testing-output")
        with cols[2]:
            st.session_state.rlcd_template_name = st.text_input(
                "Chat template", value="chatml"
            )  # TODO: 到时候import conversation把这个改成selectbox，现在暂时没时间这么处理
            st.session_state.rlcd_chosen_tag = st.text_input("Chosen source tag", value="Qwen2.5_chosen")
        with cols[3]:
            st.session_state.rlcd_rejected_tag = st.text_input("Rejected source tag", value="Qwen2.5_rejected")
        st.write("prepare for dpo")
        cols = st.columns(3)
        with cols[0]:
            st.session_state.rlcd_dpo_chosen = st.text_input("Chosen file", value="testing-output")
            st.session_state.rlcd_dpo_rejected = st.text_input("Rejected file", value="testing-output")
        with cols[1]:
            st.session_state.rlcd_dpo_chosen_tag = st.text_input(
                "Dpo chosen source tag", value="Qwen2.5_chosen"
            )
            st.session_state.rlcd_dpo_rejected_tag = st.text_input(
                "Dpo rejected source tag", value="Qwen2.5_rejected"
            )
        with cols[2]:
            st.session_state.rlcd_output_dir = st.text_input("Output dir", value="testing-output")
            st.session_state.self_re_log_dir = st.text_input("Logging dir", value="outputs")
        submit_rlcd = st.form_submit_button("→")
        if submit_rlcd:
            missing_fields = []
            # 检查文本字段是否为空
            if not st.session_state.rlcd_input_file_dir.strip():
                missing_fields.append("Prompt File")
            if not st.session_state.rlcd_output_chosen.strip():
                missing_fields.append("Output Chosen")
            if not st.session_state.rlcd_output_rejected.strip():
                missing_fields.append("Output Rejected")
            if not st.session_state.rlcd_infer_model_name.strip():
                missing_fields.append("Model name")
            if not st.session_state.rlcd_infer_model_path.strip():
                missing_fields.append("Model path")
            if not st.session_state.rlcd_infer_chosen.strip():
                missing_fields.append("Output chosen file")
            if not st.session_state.rlcd_infer_rejected.strip():
                missing_fields.append("Output rejected file")
            if not st.session_state.rlcd_template_name.strip():
                missing_fields.append("Chat template")
            if not st.session_state.rlcd_chosen_tag.strip():
                missing_fields.append("Chosen source tag")
            if not st.session_state.rlcd_rejected_tag.strip():
                missing_fields.append("Rejected source tag")
            if not st.session_state.rlcd_dpo_chosen.strip():
                missing_fields.append("Chosen file")
            if not st.session_state.rlcd_dpo_rejected.strip():
                missing_fields.append("Rejected file")
            if not st.session_state.rlcd_dpo_chosen_tag.strip():
                missing_fields.append("Dpo chosen source tag")
            if not st.session_state.rlcd_dpo_rejected_tag.strip():
                missing_fields.append("Dpo rejected source tag")
            if not st.session_state.rlcd_output_dir.strip():
                missing_fields.append("Output dir")

            # 检查路径合法性
            invalid_paths = []
            if st.session_state.rlcd_input_file_dir and not os.path.exists(st.session_state.rlcd_input_file_dir):
                invalid_paths.append(f"Prompt File '{st.session_state.rlcd_input_file_dir}' does not exist.")
            if st.session_state.rlcd_infer_model_path and not os.path.exists(st.session_state.rlcd_infer_model_path):
                invalid_paths.append(f"Model Path '{st.session_state.rlcd_infer_model_path}' does not exist.")

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
                log_dir = os.path.join(st.session_state.self_re_log_dir, "self_rewarding_log.log")
                script_content = f"""
python algorithms/rlcd_sys/prepare_for_rlcd.py --input-file {st.session_state.rlcd_input_file_dir} \
                            --output-chosen {st.session_state.rlcd_output_chosen} \
                            --output-rejected {st.session_state.rlcd_output_rejected}

autoalign-cli infer --backend "vllm" \
            --model-name {st.session_state.rlcd_infer_model_name} \
            --model-path {st.session_state.rlcd_infer_model_path} \
            --test-file1 {st.session_state.rlcd_infer_chosen} \
            --template {st.session_state.rlcd_template_name} \
            --source {st.session_state.rlcd_chosen_tag} \
            --output-dir1 {st.session_state.rlcd_dpo_chosen}

autoalign-cli infer --backend "vllm" \
            --model-name {st.session_state.rlcd_infer_model_name} \
            --model-path {st.session_state.rlcd_infer_model_path} \
            --test-file2 {st.session_state.rlcd_infer_rejected} \
            --template {st.session_state.rlcd_template_name} \
            --source {st.session_state.rlcd_rejected_tag} \
            --output-dir2 {st.session_state.rlcd_dpo_rejected}

python -m autoalign.data.prepare_for_dpo --input-files {st.session_state.rlcd_dpo_chosen} \
                                                        {st.session_state.rlcd_dpo_rejected} \
                                        --chosen-source {st.session_state.rlcd_dpo_chosen_tag} \
                                        --rejected-source {st.session_state.rlcd_dpo_rejected_tag} \
                                        --output-file-path {st.session_state.rlcd_output_dir} \
                                        --remove-system-message \
                                        --abandon-same-response 2>&1 | tee {log_dir}; echo "###page7###" >> {log_dir}
"""
                # current_dir = os.path.dirname(os.path.abspath(__file__))

                # bash_file_path = os.path.join(current_dir, "rlcd.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                st.session_state.step2 = script_content
                st.session_state.p2_fin = True
                st.session_state.selected_button = "train"
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.switch_page("pages/page3.py")
elif st.session_state.method == "SPIN":
    with st.form("input"):
        st.write("inference for spin")
        cols = st.columns(3)
        with cols[0]:
            st.session_state.spin_model_name = st.text_input("Model name")
            st.session_state.spin_model_path = st.text_input("Model path")
        with cols[1]:
            st.session_state.spin_test_file = st.text_input("Prompt file")
            st.session_state.spin_template = st.text_input("Template name")
        with cols[2]:
            st.session_state.spin_source = st.text_input("Rejected source tag")
        st.write("prepare for dpo")
        cols = st.columns(3)
        with cols[0]:
            st.session_state.spin_dpo_input_chosen = st.text_input("Chosen file")
            st.session_state.spin_dpo_input_rejected = st.text_input("Rejected file")
        with cols[1]:
            st.session_state.spin_dpo_chosen_source = st.text_input(
                "Dpo chosen source tag"
            )
            st.session_state.spin_dpo_rejected_source = st.text_input(
                "Dpo rejected source tag"
            )
        with cols[2]:
            st.session_state.spin_output_file_dir = st.text_input("Output dir")
        submit_spin = st.form_submit_button("→")
        if submit_spin:
            missing_fields = []
            # 检查文本字段是否为空
            if not st.session_state.spin_model_name.strip():
                missing_fields.append("Model name")
            if not st.session_state.spin_model_path.strip():
                missing_fields.append("Model path")
            if not st.session_state.spin_test_file.strip():
                missing_fields.append("Prompt file")
            if not st.session_state.spin_template.strip():
                missing_fields.append("Template name")
            if not st.session_state.spin_source.strip():
                missing_fields.append("Rejected source tag")
            if not st.session_state.spin_dpo_input_chosen.strip():
                missing_fields.append("Chosen file")
            if not st.session_state.spin_dpo_input_rejected.strip():
                missing_fields.append("Rejected file")
            if not st.session_state.spin_dpo_chosen_source.strip():
                missing_fields.append("Dpo chosen source tag")
            if not st.session_state.spin_dpo_rejected_source.strip():
                missing_fields.append("Dpo rejected source tag")
            if not st.session_state.spin_output_file_dir.strip():
                missing_fields.append("Output dir")

            # 检查路径合法性
            invalid_paths = []
            if st.session_state.spin_test_file and not os.path.exists(st.session_state.spin_test_file):
                invalid_paths.append(f"Test File '{st.session_state.spin_test_file}' does not exist.")
            if st.session_state.spin_model_path and not os.path.exists(st.session_state.spin_model_path):
                invalid_paths.append(f"Model Path '{st.session_state.spin_model_path}' does not exist.")

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
autoalign-cli infer --backend "vllm" \\
            --model-name {st.session_state.spin_model_name} \\
            --model-path {st.session_state.spin_model_path} \\
            --test-file {st.session_state.spin_test_file} \\
            --template {st.session_state.spin_template} \\
            --source {st.session_state.spin_source}

python -m autoalign.data.prepare_for_dpo --input-files {st.session_state.spin_dpo_input_chosen} \\
                                                        {st.session_state.spin_dpo_input_rejected} \\
                                        --chosen-source {st.session_state.spin_dpo_chosen_source} \\
                                        --rejected-source {st.session_state.spin_dpo_rejected_source} \\
                                        --output-file-path {st.session_state.spin_output_file_dir} \\
                                        --set-source-tag "0->golden" \\
                                        --remove-system-message \\
                                        --abandon-same-response
"""
                # current_dir = os.path.dirname(os.path.abspath(__file__))
                # bash_file_path = os.path.join(current_dir, "spin.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                st.session_state.step2 = script_content
                st.session_state.p2_fin = True
                st.session_state.selected_button = "train"
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.switch_page("pages/page3.py")
elif st.session_state.method == "CAI_sft":
    with st.form("input"):
        st.write("prepare for cai")
        cols = st.columns(4)
        with cols[0]:
            st.session_state.cai_sft_model_name = st.text_input("Model name", value="Qwen2.5-Ins")
            st.session_state.cai_sft_model_path = st.text_input("Model path", value="/141nfs/arknet/hf_models/Qwen1.5-1.8B-Chat/")
        with cols[1]:
            st.session_state.cai_sft_input_path = st.text_input("Input file", value="testing-output")
            st.session_state.cai_sft_input_helpful_path = st.text_input("Helpful file", value="algorithms/cai/ultra_90k.json")
        with cols[2]:
            st.session_state.cai_sft_output_chosen = st.text_input("Chosen output", value="outputs/chosen.json")
            st.session_state.cai_sft_output_rejected = st.text_input("Rejected output", value="outputs/rejected.json")
        with cols[3]:
            st.session_state.cai_sft_output_cai = st.text_input("Output CAI file name", value="outputs/cai.json")
            st.session_state.cai_sft_output_sft = st.text_input("Output sft file name", value="testing-output")
            st.session_state.self_re_log_dir = st.text_input("Logging dir", value="outputs")
        submit_cai_sft = st.form_submit_button("→")
        if submit_cai_sft:
            missing_fields = []
            # 检查文本字段是否为空
            if not st.session_state.cai_sft_model_name.strip():
                missing_fields.append("Model name")
            if not st.session_state.cai_sft_model_path.strip():
                missing_fields.append("Model path")
            if not st.session_state.cai_sft_input_path.strip():
                missing_fields.append("Input file")
            if not st.session_state.cai_sft_input_helpful_path.strip():
                missing_fields.append("Helpful file")
            if not st.session_state.cai_sft_output_chosen.strip():
                missing_fields.append("Chosen output")
            if not st.session_state.cai_sft_output_rejected.strip():
                missing_fields.append("Rejected output")
            if not st.session_state.cai_sft_output_cai.strip():
                missing_fields.append("Output CAI file name")
            if not st.session_state.cai_sft_output_sft.strip():
                missing_fields.append("Output sft file name")

            # 检查路径合法性
            invalid_paths = []
            if st.session_state.cai_sft_input_path and not os.path.exists(st.session_state.cai_sft_input_path):
                invalid_paths.append(f"Input File '{st.session_state.cai_sft_input_path}' does not exist.")
            if st.session_state.cai_sft_input_helpful_path and not os.path.exists(st.session_state.cai_sft_input_helpful_path):
                invalid_paths.append(f"Helpful File '{st.session_state.cai_sft_input_helpful_path}' does not exist.")
            if st.session_state.cai_sft_model_path and not os.path.exists(st.session_state.cai_sft_model_path):
                invalid_paths.append(f"Model Path '{st.session_state.cai_sft_model_path}' does not exist.")

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
                log_dir = os.path.join(st.session_state.self_re_log_dir, "self_rewarding_log.log")
                script_content = f"""
python algorithms/cai/change.py --file {st.session_state.cai_sft_input_path}

python algorithms/cai/prepare_for_cai.py   --model-name {st.session_state.cai_sft_model_name} \
                            --model-path {st.session_state.cai_sft_model_path} \
                            --input-file {st.session_state.cai_sft_input_path} \
                            --input_helpful_file {st.session_state.cai_sft_input_helpful_path} \
                            --output-chosen {st.session_state.cai_sft_output_chosen} \
                            --output-rejected {st.session_state.cai_sft_output_rejected} \
                            --output-cai {st.session_state.cai_sft_output_cai} \
                            --output-sft {st.session_state.cai_sft_output_sft} 2>&1 | tee {log_dir}; echo "###page7###" >> {log_dir}
"""
                # current_dir = os.path.dirname(os.path.abspath(__file__))
                # # current_dir = "/141nfs/wangpengbo/auto_alignment/auto-alignment/algorithms/cai/"
                # bash_file_path = os.path.join(current_dir, "prepare_for_cai.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                st.session_state.step2 = script_content
                st.session_state.p2_fin = True
                st.session_state.selected_button = "train"
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.switch_page("pages/page3.py")
elif st.session_state.method == "CAI_dpo":
    with st.form("input"):
        cols = st.columns(2)
        with cols[0]:
            st.session_state.cai_dpo_model_name = st.text_input("Model name", value="Qwen2.5-Ins")
            st.session_state.cai_dpo_model_path = st.text_input("Model path", value="/141nfs/arknet/hf_models/Qwen1.5-1.8B-Chat/")
        with cols[1]:
            st.session_state.cai_dpo_input_path = st.text_input("Input file", value="testing-output")
            st.session_state.cai_dpo_output_path = st.text_input("Output file", value="testing-output")
            st.session_state.self_re_log_dir = st.text_input("Logging dir", value="outputs")
        submit_cai_dpo = st.form_submit_button("→")
        if submit_cai_dpo:
            missing_fields = []
            # 检查文本字段是否为空
            if not st.session_state.cai_dpo_model_name.strip():
                missing_fields.append("Model name")
            if not st.session_state.cai_dpo_model_path.strip():
                missing_fields.append("Model path")
            if not st.session_state.cai_dpo_input_path.strip():
                missing_fields.append("Input file")
            if not st.session_state.cai_dpo_output_path.strip():
                missing_fields.append("Output file")

            # 检查路径合法性
            invalid_paths = []
            if st.session_state.cai_dpo_input_path and not os.path.exists(st.session_state.cai_dpo_input_path):
                invalid_paths.append(f"Input File '{st.session_state.cai_dpo_input_path}' does not exist.")
            if st.session_state.cai_dpo_model_path and not os.path.exists(st.session_state.cai_dpo_model_path):
                invalid_paths.append(f"Model Path '{st.session_state.cai_dpo_model_path}' does not exist.")

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
                log_dir = os.path.join(st.session_state.self_re_log_dir, "self_rewarding_log.log")
                script_content = f"""
    python algorithms/cai/temperature_sample.py   --model-name {st.session_state.cai_dpo_model_name} \
                                --model-path {st.session_state.cai_dpo_model_path}/checkpoint-* \
                                --input-file {st.session_state.cai_dpo_input_path} \
                                --output-file {st.session_state.cai_dpo_output_path} 2>&1 | tee {log_dir}; echo "###page7###" >> {log_dir}
    """
                # current_dir = os.path.dirname(os.path.abspath(__file__))
                # # current_dir = "/141nfs/wangpengbo/auto_alignment/auto-alignment/algorithms/cai/"
                # bash_file_path = os.path.join(current_dir, "temperature_sample.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                st.session_state.step2 = script_content
                st.session_state.p2_fin = True
                st.session_state.selected_button = "train"
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.switch_page("pages/page3.py")
elif st.session_state.method == "Self_Rewarding":
    with st.form("input"):
        cols = st.columns(3)
        with cols[0]:
            st.session_state.self_re_model_path = st.text_input("Model path", value="/141nfs/arknet/hf_models/Qwen1.5-1.8B-Chat/")
            st.session_state.self_re_model_id = st.text_input("Model id", value="Qwen2.5-Ins")
            st.session_state.self_re_template_name = st.selectbox(
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
        with cols[1]:
            st.session_state.self_re_sft_base_model = st.selectbox("SFT base model", ("eft", "ift"))
            st.session_state.self_re_backend = st.text_input("Backend", value="vllm")
            st.session_state.output_path = st.text_input("Output Path", value="testing-output")
        with cols[2]:
            st.session_state.self_re_num_iter = st.text_input("Num iter", value=1)
            st.session_state.self_re_ins_path = st.text_input("Instruction path", value="testing-output")
            st.session_state.self_re_log_dir = st.text_input("Logging dir", value="outputs")
        submit_self_re = st.form_submit_button("→")
        if submit_self_re:
            missing_fields = []
            # 检查文本字段是否为空
            if not st.session_state.self_re_model_id.strip():
                missing_fields.append("Model id")
            if not st.session_state.self_re_model_path.strip():
                missing_fields.append("Model path")
            if not st.session_state.self_re_template_name.strip():
                missing_fields.append("Template name")
            if not st.session_state.self_re_sft_base_model.strip():
                missing_fields.append("SFT base model")
            if not st.session_state.self_re_backend.strip():
                missing_fields.append("Backend")
            if not st.session_state.self_re_num_iter.strip():
                missing_fields.append("Num iter")
            if not st.session_state.self_re_ins_path.strip():
                missing_fields.append("Instruction path")
            if not st.session_state.self_re_log_dir.strip():
                missing_fields.append("Log path")
            
            # 检查路径合法性
            invalid_paths = []
            if st.session_state.self_re_ins_path and not os.path.exists(st.session_state.self_re_ins_path):
                invalid_paths.append(f"Input File '{st.session_state.self_re_ins_path}' does not exist.")
            if st.session_state.self_re_model_path and not os.path.exists(st.session_state.self_re_model_path):
                invalid_paths.append(f"Model Path '{st.session_state.self_re_model_path}' does not exist.")

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
                log_dir = os.path.join(st.session_state.self_re_log_dir, "self_rewarding_log.log")
                script_content = f"""
python algorithms/self-rewarding/src/dpo_dataset_generator.py    --model-path {st.session_state.self_re_model_path} \
                                       --model-id {st.session_state.self_re_model_id} \
                                       --template-name {st.session_state.self_re_template_name} \
                                       --sft-base-model {st.session_state.self_re_sft_base_model} \
                                       --backend {st.session_state.self_re_backend} \
                                       --num-iter {st.session_state.self_re_num_iter} \
                                       --instruction-path {st.session_state.self_re_ins_path} \
                                       --output-path {st.session_state.output_path} 2>&1 | tee {log_dir}; echo "###page7###" >> {log_dir}
"""
                st.session_state.step2 = script_content
                st.session_state.p2_fin = True
                st.session_state.selected_button = "train"
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.switch_page("pages/page3.py")


if st.session_state.selected_button == "data_gen":
    st.switch_page("pages/page1.py")
elif st.session_state.selected_button == "data_filter":
    pass
elif st.session_state.selected_button == "train":
    if st.session_state.p1_fin and st.session_state.p2_fin:
        st.switch_page("pages/page3.py")
    else:
        st.title("Please finish this page first")
        st.session_state.selected_button == "data_filter"
elif st.session_state.selected_button == "eval":
    if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.switch_page("pages/page4.py")
