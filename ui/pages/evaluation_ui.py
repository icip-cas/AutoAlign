import streamlit as st
import os
import time
from pages.navigation import render_navbar, init_session_state
import subprocess

render_navbar()
hide_sidebar_css = """
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_sidebar_css, unsafe_allow_html=True)
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
        # value=st.session_state["model_name"],  # 使用 session_state 中的值
        value="aaa",
        label_visibility="collapsed",
    )

    #输出地址
    st.subheader("📄 Output Path")
    output_path = st.text_input(
        "配置文件存储路径",
        placeholder="请输入配置存储路径",
        value = "ata-output",
        label_visibility="collapsed"
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
    submitted = st.form_submit_button("🚀")

    # 表单提交后的逻辑
    if submitted:
        # 保存用户输入到 session_state
        st.session_state["process"] = process
        st.session_state["model_dir"] = model_dir
        st.session_state["model_name"] = model_name
        st.session_state["per_model_gpu"] = per_model_gpu
        st.session_state["batch_size"] = batch_size
        st.session_state["output_dir"] = output_path

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
            config_content = f"""
model_name: {model_name}
template_name: chatml-keep-system
model_path: {model_dir}
eval_type: {process}
per_model_gpu: {per_model_gpu}
batch_size: {batch_size}
backend: vllm
opencompass_path: opencompass
mt_path: {mt_path}
judge_model: chatgpt_fn
"""

            if alpaca_path:
                config_content += f"""
alpaca_path: {alpaca_path}
"""
            st.session_state.step4 = config_content
            st.session_state.p4_fin = True
            st.success(f"Align Start!")
            time.sleep(1.5)
            #  TODO: 在这里插入执行一次性脚本的内容
            os.environ['step1'] = st.session_state.step1
            os.environ['step2'] = st.session_state.step2
            os.environ['step3'] = st.session_state.step3
            os.environ['step4'] = st.session_state.step4
            os.environ['syn_method'] = st.session_state.Syn_method
            os.environ['method'] = st.session_state.method
            os.environ['epoch'] = str(st.session_state.total_epoch)
            os.environ['eval_path'] = st.session_state.output_dir
            subprocess.Popen(f"python ui/pages/iteration_setting.py 2>&1 | tee outputs/rec.log", text=True, shell=True)
            # subprocess.Popen("python ui/pages/Align.py", text=True, shell=True)
            st.switch_page("pages/loading_pages_ui.py")

if st.session_state.selected_button == "data_gen":
    st.switch_page("pages/instruction_generation_ui.py")
elif st.session_state.selected_button == "data_filter":
    st.switch_page("pages/sampling_answer_ui.py")
elif st.session_state.selected_button == "train":
    st.switch_page("pages/training_ui.py")
elif st.session_state.selected_button == "eval":
    pass
