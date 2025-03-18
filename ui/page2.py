import streamlit as st
import os
import time
for key, default in {
    "p1_fin": False,
    "p2_fin": False,
    "p3_fin": False,
    "p4_fin": False,
    "selected_button": "data_gen"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        div.stButton > button,
        div.stFormSubmitButton > button {
            width: min(6vw, 80px);
            height: min(6vw, 80px);
            border-radius: 50%;
            background: #2196F3;
            color: white !important;  /* 强制文字颜色 */
            border: none;
            cursor: pointer;
            transition: 0.3s;
            font-size: 2rem !important;
            font-weight: bold !important;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
        }
            
        div.stButton > button > div > p,
        div.stFormSubmitButton > button > div > p {
            font-size: 2rem;    
        }

        /* 覆盖所有交互状态 */
        div.stButton > button:hover,
        div.stButton > button:active,
        div.stButton > button:focus,
        div.stFormSubmitButton > button:hover,
        div.stFormSubmitButton > button:active,
        div.stFormSubmitButton > button:focus {
            background: #1976D2 !important;
            color: white !important;  /* 强制保持白色 */
            transform: scale(1.05);
            box-shadow: none !important;  /* 移除聚焦阴影 */
            outline: none !important;     /* 移除聚焦轮廓 */
        }

        /* 强制禁用所有颜色变换 */
        div.stButton > button:hover span,
        div.stButton > button:active span,
        div.stButton > button:focus span {
            color: inherit !important;  /* 继承父级颜色 */
        }
        
        .btn-text {
            font-size: 1.5rem;
            text-align: center;
            margin-top: 5px;
        }
        
    </style>
""", unsafe_allow_html=True)


page_layout = st.columns([1, 5, 1])
with page_layout[1]:
    nav_cols = st.columns(7)
    labels = [
        ("Generate Query", "1", "data_gen"),
        ("", "", ""),
        ("Sample Answer", "2", "data_filter"),
        ("", "", ""),
        ("Train", "3", "train"),
        ("", "", ""),
        ("Eval", "4", "eval")
    ]


    for i, (text, num, key) in enumerate(labels):
        with nav_cols[i]:
            if key:
                with st.container():
                    if st.button(num, key=key):
                        st.session_state['selected_button'] = key
                    st.markdown(f"<div class='btn-text'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='hr-line' style='background: #2196F3; height: 4px; width: 100%; margin: 39px auto 0; border-radius: 2px;'></div>", unsafe_allow_html=True)


    st.session_state.method = st.selectbox(
        "Sample Method",
        ("RLCD", "SPIN", "CAI_sft", "CAI_dpo", "Self_Rewarding")
    )
    if st.session_state.method == "RLCD":
        with st.form('input'):
            st.write("prepare for rlcd")
            cols = st.columns(3)
            with cols[0]:
                st.session_state.rlcd_input_file_dir = st.text_input("Prompt File")
            with cols[1]:
                st.session_state.rlcd_output_chosen = st.text_input("Output Chosen")
            with cols[2]:
                st.session_state.rlcd_output_rejected = st.text_input("Output Rejected")
            st.write("inference for rlcd")
            cols = st.columns(4)
            with cols[0]:
                st.session_state.rlcd_infer_model_name = st.text_input("Model name")
                st.session_state.rlcd_infer_model_path = st.text_input("Model path")
            with cols[1]:
                st.session_state.rlcd_infer_chosen = st.text_input("Output chosen file")
                st.session_state.rlcd_infer_rejected = st.text_input("Output rejected file")
            with cols[2]:    
                st.session_state.rlcd_template_name = st.text_input("Chat template") # TODO: 到时候import conversation把这个改成selectbox，现在暂时没时间这么处理
                st.session_state.rlcd_chosen_tag = st.text_input("Chosen source tag")
            with cols[3]:
                st.session_state.rlcd_rejected_tag = st.text_input("Rejected source tag")
            st.write("prepare for dpo")
            cols = st.columns(3)
            with cols[0]:
                st.session_state.rlcd_dpo_chosen = st.text_input("Chosen file")
                st.session_state.rlcd_dpo_rejected = st.text_input("Rejected file")
            with cols[1]:    
                st.session_state.rlcd_dpo_chosen_tag = st.text_input("Dpo chosen source tag")
                st.session_state.rlcd_dpo_rejected_tag = st.text_input("Dpo rejected source tag")
            with cols[2]:    
                st.session_state.rlcd_output_dir = st.text_input("Output dir")
            submit_rlcd = st.form_submit_button('→')
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
                if missing_fields:
                    st.error(f"Missing or invalid fields: {', '.join(missing_fields)}")
                else:
                    script_content = f"""
python prepare_for_rlcd.py --input-file {st.session_state.rlcd_input_file_dir} \
                            --output-chosen {st.session_state.rlcd_output_chosen} \
                            --output-rejected {st.session_state.rlcd_output_rejected}

autoalign-cli infer --backend "vllm" \
            --model-name {st.session_state.rlcd_infer_model_name} \
            --model-path {st.session_state.rlcd_infer_model_path} \
            --test-file {st.session_state.rlcd_output_chosen} \
            --template {st.session_state.rlcd_template_name} \
            --source {st.session_state.rlcd_chosen_tag}

autoalign-cli infer --backend "vllm" \
            --model-name {st.session_state.rlcd_infer_model_name} \
            --model-path {st.session_state.rlcd_infer_model_path} \
            --test-file {st.session_state.rlcd_output_rejected} \
            --template {st.session_state.rlcd_template_name} \
            --source {st.session_state.rlcd_rejected_tag}

python -m autoalign.data.prepare_for_dpo --input-files {st.session_state.rlcd_dpo_chosen} \
                                                        {st.session_state.rlcd_dpo_rejected} \
                                        --chosen-source {st.session_state.rlcd_dpo_chosen_tag} \
                                        --rejected-source {st.session_state.rlcd_dpo_rejected_tag} \
                                        --output-file-path {st.session_state.rlcd_output_dir} \
                                        --remove-system-message \
                                        --abandon-same-response
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
                    st.switch_page("page3.py")
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
                st.session_state.spin_dpo_chosen_source = st.text_input("Dpo chosen source tag")
                st.session_state.spin_dpo_rejected_source = st.text_input("Dpo rejected source tag")
            with cols[2]:
                st.session_state.spin_output_file_dir = st.text_input("Output dir")
            submit_spin = st.form_submit_button('→')
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
                if missing_fields:
                    st.error(f"Missing or invalid fields: {', '.join(missing_fields)}")
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
                    st.switch_page("page3.py")
    elif st.session_state.method == "CAI_sft":
        with st.form("input"):
            st.write("prepare for cai")
            cols = st.columns(4)
            with cols[0]:
                st.session_state.cai_sft_model_name = st.text_input("Model name")
                st.session_state.cai_sft_model_path = st.text_input("Model path")
            with cols[1]:
                st.session_state.cai_sft_input_path = st.text_input("Input file")
                st.session_state.cai_sft_output_path = st.text_input("Output file")
            with cols[2]:
                st.session_state.cai_sft_output_chosen = st.text_input("Chosen output")
                st.session_state.cai_sft_output_rejected = st.text_input("Rejected output")
            with cols[3]:
                st.session_state.cai_sft_output_cai = st.text_input("Output CAI file name")
                st.session_state.cai_sft_output_sft = st.text_input("Output sft file name")
            submit_cai_sft = st.form_submit_button('→')
            if submit_cai_sft:
                missing_fields = []
                # 检查文本字段是否为空
                if not st.session_state.cai_sft_model_name.strip():
                    missing_fields.append("Model name")
                if not st.session_state.cai_sft_model_path.strip():
                    missing_fields.append("Model path")
                if not st.session_state.cai_sft_input_path.strip():
                    missing_fields.append("Input file")
                if not st.session_state.cai_sft_output_path.strip():
                    missing_fields.append("Output file")
                if not st.session_state.cai_sft_output_chosen.strip():
                    missing_fields.append("Chosen output")
                if not st.session_state.cai_sft_output_rejected.strip():
                    missing_fields.append("Rejected output")
                if not st.session_state.cai_sft_output_cai.strip():
                    missing_fields.append("Output CAI file name")
                if not st.session_state.cai_sft_output_sft.strip():
                    missing_fields.append("Output sft file name")
                if missing_fields:
                    st.error(f"Missing or invalid fields: {', '.join(missing_fields)}")
                else:
                    script_content = f"""
python prepare_for_cai.py   --model-name {st.session_state.cai_sft_model_name} \
                            --model-path {st.session_state.cai_sft_model_path} \
                            --input-file {st.session_state.cai_sft_input_path} \
                            --input_helpful_file {st.session_state.cai_sft_output_path} \
                            --output-chosen {st.session_state.cai_sft_output_chosen} \
                            --output-rejected {st.session_state.cai_sft_output_rejected} \
                            --output-cai {st.session_state.cai_sft_output_cai} \
                            --output-sft {st.session_state.cai_sft_output_sft}
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
                    st.switch_page("page3.py")
    elif st.session_state.method == "CAI_dpo":
        with st.form("input"):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.cai_dpo_model_name = st.text_input("Model name")
                st.session_state.cai_dpo_model_path = st.text_input("Model path")
            with cols[1]:
                st.session_state.cai_dpo_input_path = st.text_input("Input file")
                st.session_state.cai_dpo_output_path = st.text_input("Output file")
            submit_cai_dpo = st.form_submit_button('→')
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
                if missing_fields:
                    st.error(f"Missing or invalid fields: {', '.join(missing_fields)}")
                else:
                    script_content = f"""
    python temperature_sample.py   --model-name {st.session_state.cai_dpo_model_name} \
                                --model-path {st.session_state.cai_dpo_model_path}/checkpoint-* \
                                --input-file {st.session_state.cai_dpo_input_path} \
                                --output-file {st.session_state.cai_dpo_output_path}
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
                    st.switch_page("page3.py")
    elif st.session_state.method == "Self_Rewarding":
        with st.form("input"):
            cols = st.columns(3)
            with cols[0]:
                st.session_state.self_re_model_path = st.text_input("Model path")
                st.session_state.self_re_model_id = st.text_input("Model id")
                st.session_state.self_re_template_name = st.text_input("Template name")
            with cols[1]:
                st.session_state.self_re_sft_base_model = st.text_input("SFT base model")
                st.session_state.self_re_backend = st.text_input("Backend")
            with cols[2]:
                st.session_state.self_re_num_iter = st.text_input("Num iter")
                st.session_state.self_re_ins_path = st.text_input("Instruction path")
            submit_self_re = st.form_submit_button('→')
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
                if missing_fields:
                    st.error(f"Missing or invalid fields: {', '.join(missing_fields)}")
                else:
                    script_content = f"""
python src/dpo_dataset_generator.py    --model-path {st.session_state.self_re_model_path} \
                                       --model-id {st.session_state.self_re_model_id} \
                                       --template-name {st.session_state.self_re_template_name} \
                                       --sft-base-model {st.session_state.self_re_sft_base_model} \
                                       --backend {st.session_state.self_re_backend} \
                                       --num-iter {st.session_state.self_re_num_iter} \
                                       --instruction-path {st.session_state.self_re_ins_path}
"""
                    # current_dir = os.path.dirname(os.path.abspath(__file__))
                    # bash_file_path = os.path.join(current_dir, "self_rewarding.sh")
                    # # 将脚本内容保存到文件
                    # with open(bash_file_path, "w") as f:
                    #     f.write(script_content)
                    st.session_state.step2 = script_content
                    st.session_state.p2_fin = True
                    st.success("Configuration Saved!")
                    time.sleep(1.5)
                    st.switch_page("page3.py")
            







    if st.session_state.selected_button == "data_gen":
        st.switch_page("page1.py")
    elif st.session_state.selected_button == "data_filter":
        pass
    elif st.session_state.selected_button == "train":
        if st.session_state.p1_fin and st.session_state.p2_fin:
            st.switch_page("page3.py")
        else:
            st.title("Please finish this page first")
            st.session_state.selected_button == "data_filter"
    elif st.session_state.selected_button == "eval":
        if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
            st.switch_page("page4.py")