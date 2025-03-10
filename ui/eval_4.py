import streamlit as st
import os

st.title("Model Eval")

# 初始化 session_state
if "process" not in st.session_state:
    st.session_state["process"] = "MATH"
if "model_dir" not in st.session_state:
    st.session_state["model_dir"] = ""
if "output_dir" not in st.session_state:
    st.session_state["output_dir"] = ""

# 配置表单
with st.form("config_form"):
    st.subheader("BenchMark")
    col1, col2 = st.columns([3, 2])
    with col1:
        process = st.selectbox(
            "BenchMark",
            [
                "MATH",
                "GSM-8K",
                "HumanEval",
                "MBPP",
                "HumanEval-CN",
                "MBPP-CN",
                "MMLU",
                "GPQA",
                "CMMLU",
                "C-Eval",
            ],
            index=[
                "MATH",
                "GSM-8K",
                "HumanEval",
                "MBPP",
                "HumanEval-CN",
                "MBPP-CN",
                "MMLU",
                "GPQA",
                "CMMLU",
                "C-Eval",
            ].index(st.session_state["process"]),  # 恢复 process 的选择状态
            label_visibility="collapsed"
        )

    st.subheader("Model Selection")
    model_dir = st.text_input(
        "Model Dir", 
        placeholder="Please provide the path for the model.", 
        value=st.session_state["model_dir"],  # 恢复 model_dir 的输入内容
        label_visibility="collapsed"
    )

    st.subheader("Output Path")
    output_dir = st.text_input(
        "Result Dir", 
        placeholder="Please specify the path for saving the results.", 
        value=st.session_state["output_dir"],  # 恢复 output_dir 的输入内容
        label_visibility="collapsed"
    )

    # 提交按钮
    col1, col2, col3 = st.columns([4, 2, 4])
    with col2:
        submitted = st.form_submit_button("🚀 Start")

    # 表单提交后的逻辑
    if submitted:
        # 保存用户输入到 session_state
        st.session_state["process"] = process
        st.session_state["model_dir"] = model_dir
        st.session_state["output_dir"] = output_dir

        all_fields_filled = True

        # 检查 Model Dir 是否存在
        if not model_dir:
            st.error("Please provide the model directory.")
            all_fields_filled = False
        elif not os.path.exists(model_dir):
            st.error(f"Model directory '{model_dir}' does not exist.")
            all_fields_filled = False

        # 检查 Output Dir 是否存在
        if not output_dir:
            st.error("Please specify the output directory.")
            all_fields_filled = False

        # 如果所有字段合法且路径检查通过
        if all_fields_filled:
            # 检查 Output Dir 是否存在，如果不存在则创建
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)  # 尝试创建目录
                    st.success(f"Output directory '{output_dir}' created successfully.")
                except Exception as e:
                    st.error(f"Failed to create output directory '{output_dir}': {e}")
                    all_fields_filled = False

            # 如果创建成功，跳转到 page5.py
            if all_fields_filled:
                st.switch_page("page5.py")