import streamlit as st
import pandas as pd
import time
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("Data Board", use_container_width=True):
        st.session_state.selected_button = "data_demo"
with nav_cols[1]:
    if st.button("Logs", use_container_width=True):
        st.session_state.selected_button = "logs"
with nav_cols[2]:
    if st.button("Training", use_container_width=True) and st.session_state.p2_fin:
        st.session_state.selected_button = "training"
with nav_cols[3]:
    if st.button("Benchmark", use_container_width=True) and st.session_state.p2_fin and st.session_state.p3_fin:
        st.session_state.selected_button = "benchmark"

st.title("Benchmark Results")

st.markdown("---")  

st.header("The evaluation progress of the benchmark")


progress_value = 0
progress_bar = st.progress(progress_value) 
status_text = st.empty()  

# 模拟进度更新
for i in range(101):
    time.sleep(0.02) 
    progress_bar.progress(i)  
    status_text.text(f"Current Progress: {i}%")  

# 任务完成提示
st.success("The benchmark evaluation is complete！")

# 添加一些空白空间，使进度条区域更大
st.markdown("<br>", unsafe_allow_html=True)  # 使用 HTML 换行增加空间

# 下半部分：多列两行的空表格
st.header("Result")

columns = [
    "GSM-8K", "HumanEval", "MBPP", "HumanEval-CN",
    "MBPP-CN", "MMLU", "GPQA", "CMMLU", "C-Eval"
]

data = {col: [""] for col in columns}  
df = pd.DataFrame(data)

df.index = ["score"]  

st.dataframe(df, height=50)  # 设置表格高度

# 添加一些样式美化界面
st.markdown("""
<style>
    /* 调整进度条的高度和颜色 */
    .stProgress > div > div > div {
        height: 25px;
        background-color: #4CAF50; /* 绿色进度条 */
    }
    /* 调整表格的边框和字体 */
    .stDataFrame {
        border: 1px solid #ccc;
        border-radius: 5px;
        font-size: 16px;
    }
    /* 调整任务完成提示的样式 */
    .stSuccess {
        font-size: 18px;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.selected_button == "data_demo":
    st.switch_page("page5.py")
elif st.session_state.selected_button == "logs":
    st.switch_page("page6.py")
elif st.session_state.selected_button == "training":
    st.switch_page("page7.py")
elif st.session_state.selected_button == "benchmark":
    pass