import streamlit as st
import os

st.title("📊 Model Evaluation")

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


process_descriptions = {
    "objective_core": "GSM-8K(EN)，MATH(EN)，HumanEval(EN)，HumanEval-CN(CH)，BBH(EN)，IFEval(EN)，CONFIG_ALL：",
    "objective_all": "GSM-8K(EN)，MATH(EN)，HumanEval(EN)HumanEval-CN(CH)，BBH(EN)，IFEval(EN)，CMMLU(CH)，C-Eval(CH)，MBPP(EN)，MBPP-CN(CH)，GPQA(EN)",
    "subjective": "MT-Bench and Alpaca-Eval",
}

# 美化标题和分隔线
st.markdown("---")
st.subheader("📊 BenchMark")

# 在表单外使用 st.selectbox，以便使用 on_change 回调
process = st.selectbox(
    "选择评测类型",
    options=[
        "objective_core",
        "objective_all",
        "subjective",
    ],
    index=[
        "objective_core",
        "objective_all",
        "subjective",
    ].index(st.session_state["process"]),  # 恢复 process 的选择状态
    key="process_selectbox",
    on_change=lambda: st.session_state.update(
        {"process": process}
    ),  # 更新 session_state
)

# 根据 session_state 中的 process 动态显示详细信息
st.markdown(
    f"""
    <style>
    .info-box {{
        padding: 15px;
        background-color: #f0f2f6;
        border-radius: 10px;
        border-left: 5px solid #4a90e2;
        margin-top: 10px;
        margin-bottom: 20px;
    }}
    .info-box p {{
        margin: 0;
        font-size: 14px;
        color: #333;
    }}
    </style>
    <div class="info-box">
        <p>📌 <strong>Benchmark:</strong> {process_descriptions.get(st.session_state["process"], "暂无描述")}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 配置表单
with st.form("config_form"):
    st.markdown("---")
    st.subheader("📂 Model Selection")

    # 模型路径输入
    model_dir = st.text_input(
        "模型路径",
        placeholder="请输入模型路径",
        value=st.session_state["model_dir"],  # 恢复 model_dir 的输入内容
        label_visibility="collapsed",
    )

    # 评测的模型标识名称
    st.subheader("🏷️ Model Name")
    model_name = st.text_input(
        "模型名称",
        placeholder="请输入模型标识名称",
        value=st.session_state["model_name"],  # 使用 session_state 中的值
        label_visibility="collapsed",
    )

    # GPU 和 Batch Size 配置
    st.markdown("---")
    st.subheader("⚙️ GPU and Batch Size Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**GPU per Model**")
        per_model_gpu = st.number_input(
            "每个模型的 GPU 数量",
            min_value=1,
            value=st.session_state["per_model_gpu"],
            step=1,
            label_visibility="collapsed",
        )
    with col2:
        st.markdown("**Batch Size**")
        batch_size = st.number_input(
            "批量大小",
            min_value=1,
            value=st.session_state["batch_size"],  # 恢复 batch_size 的输入内容
            step=1,
            label_visibility="collapsed",
        )

    # 提交按钮
    st.markdown("---")
    submitted = st.form_submit_button("🚀 Start Evaluation")

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
            st.error("❌ 请提供模型路径。")
            all_fields_filled = False
        elif not os.path.exists(model_dir):
            st.error(f"❌ 模型路径 '{model_dir}' 不存在。")
            all_fields_filled = False

        # 检查 Model Name 是否为空
        if not model_name:
            st.error("❌ 请提供模型名称。")
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
                st.success("✅ 配置文件已成功保存！")
            except Exception as e:
                st.error(f"❌ 保存配置文件失败: {e}")

            # 跳转到 page5.py
            st.switch_page("page5.py")
