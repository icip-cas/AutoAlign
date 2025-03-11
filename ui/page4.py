import streamlit as st
import os
import time
st.set_page_config(layout="centered")
nav_cols = st.columns(4)
for key, default in {
    "p1_fin": False,
    "p2_fin": False,
    "p3_fin": False,
    "p4_fin": False,
    "selected_button": "data_gen"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with nav_cols[0]:
    if st.button("📥 Generate Query", use_container_width=True):
        st.session_state.selected_button = "data_gen"
with nav_cols[1]:
    if st.button("🔎 Sample Answer", use_container_width=True):
        st.session_state.selected_button = "data_filter"
with nav_cols[2]:
    if st.button("🎓 Train", use_container_width=True) and st.session_state.p2_fin:
        st.session_state.selected_button = "train"
with nav_cols[3]:
    if st.button("📊 Eval", use_container_width=True) and st.session_state.p2_fin and st.session_state.p3_fin:
        st.session_state.selected_button = "eval"

st.title("Model Eval")

# 初始化 session_state
if "process" not in st.session_state:
    st.session_state["process"] = "objective_all"
if "model_dir" not in st.session_state:
    st.session_state["model_dir"] = ""
if "output_dir" not in st.session_state:
    st.session_state["output_dir"] = ""
if "model_name" not in st.session_state:
    st.session_state["model_name"] = ""  # 初始化为空字符串
if "per_model_gpu" not in st.session_state:
    st.session_state["per_model_gpu"] = 1
if "batch_size" not in st.session_state:
    st.session_state["batch_size"] = 8

# 配置表单
with st.form("config_form"):
    st.subheader("BenchMark")
    col1, col2 = st.columns([3, 2])
    with col1:
        process = st.selectbox(
            "BenchMark",
            [
                "objective_core",
                "objective_all",
                "subjective",
            ],
            index=[
                "objective_core",
                "objective_all",
                "subjective",
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

    # 评测的模型标识名称
    st.subheader("Model Name")
    model_name = st.text_input(
        "Model Name", 
        placeholder="Enter the identifying name of the model to evaluate.", 
        value=st.session_state["model_name"],  # 使用 session_state 中的值
        label_visibility="collapsed"
    )

    # GPU 和 Batch Size 配置
    st.subheader("GPU and Batch Size Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**GPU per Model**")
        per_model_gpu = st.number_input(
            "GPU per Model", 
            min_value=1, 
            value=st.session_state["per_model_gpu"],
            step=1,
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("**Batch Size**")
        batch_size = st.number_input(
            "Batch Size", 
            min_value=1, 
            value=st.session_state["batch_size"],  # 恢复 batch_size 的输入内容
            step=1,
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
        st.session_state["model_name"] = model_name
        st.session_state["per_model_gpu"] = per_model_gpu
        st.session_state["batch_size"] = batch_size

        all_fields_filled = True

        # 检查 Model Dir 是否存在
        if not model_dir:
            st.error("Please provide the model directory.")
            all_fields_filled = False
        elif not os.path.exists(model_dir):
            st.error(f"Model directory '{model_dir}' does not exist.")
            all_fields_filled = False

        # 检查 Model Name 是否为空
        if not model_name:
            st.error("Please provide the model name.")
            all_fields_filled = False

        # 如果所有字段合法且路径检查通过
        if all_fields_filled:
            # 根据评测类型设置 mt_path 和 alpaca_path
            mt_path = "data/eval/mt-bench"
            alpaca_path = "data/eval/alpaca_eval" if process == "subjective" else None

            # 生成配置文件内容
            config_content = f"""# 评测的模型标识名称
# Identifying name of the model to evaluate
model_name: {model_name}

# 评测时使用的上下文模板，可见src/autoalign/conversation.py中的TEMPLATES
template_name: chatml-keep-system
# 评测的模型路径
# The path of the model to evaluate
model_path: {model_dir}
# 评测的类型
# The type of evaluation
# 可选项：
# objective_core: 评测模型的核心客观指标，是objective_all对应指标的真子集。(evaluating the core objective metrics, a subset of the metrics in objective_all, of the model)
# objective_all: 评测模型的所有客观指标。(evaluating all the objective metrics of the model)
# subjective: 评测模型的主观指标。(evaluating the subjective metrics of the model)
eval_type: {process}
# 单个模型 worker 所占用的GPU数量
# The number of GPUs occupied by a single model worker
per_model_gpu: {per_model_gpu}

# 单个 worker 的 batch_size
# The batch size of a single worker
batch_size: {batch_size}

# 推理 backend
# The inference backend
backend: vllm

# ==============Opencompass 设置================
# opencompass文件夹的路径
# The path of opencompass
opencompass_path: opencompass

# ==============MTbench 设置================
# mtbench文件夹的路径
mt_path: {mt_path}

# Recommend using: chatgpt_fn or weighted_alpaca_eval_gpt4_turbo
# use weighted_alpaca_eval_gpt4_turbo if you want the high agreement with humans.
# use chatgpt_fn if you are on a tight budget.
judge_model: chatgpt_fn
"""

            if alpaca_path:
                config_content += f"""
# ==============AlpacaEval 设置================
# see https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md
# 指定AlpacaEval文件的路径(setting the alpaca eval file path if you have already downloaded it)
alpaca_path: {alpaca_path}
"""

            current_dir = os.path.dirname(__file__)

            relative_path = os.path.join("..", "configs") 
            target_dir = os.path.normpath(os.path.join(current_dir, relative_path))

            target_file = os.path.join(target_dir, "eval.yaml")
            try:
                with open(target_file, "w") as f:
                    f.write(config_content)
                st.success(f"The configuration has been successfully saved.")
                time.sleep(1.5)
                st.switch_page("page5.py")
            except Exception as e:
                st.error(f"Failed to save configuration")

if st.session_state.selected_button == "data_gen":
    st.switch_page("page1.py")
elif st.session_state.selected_button == "data_filter":
    st.switch_page("page2.py")
elif st.session_state.selected_button == "train":
    st.switch_page("page3.py")
elif st.session_state.selected_button == "eval":
    pass