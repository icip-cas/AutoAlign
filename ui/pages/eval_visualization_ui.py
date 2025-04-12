import streamlit as st
import pandas as pd
import time
from pages.navigation import render_navbar_visual
st.set_page_config(layout="wide", page_title="Real-Time Log Viewer", page_icon="📄")
hide_sidebar_css = """
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_sidebar_css, unsafe_allow_html=True)
render_navbar_visual()
# 使用缓存读取日志文件，缓存 0.5 秒
@st.cache_data(ttl=0.5)
def read_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    return log_content

# Streamlit 应用
def main():
    st.title("📄 Eval Log Viewer")
    
    log_file_path = "outputs/eval.log"  # 日志文件路径

    # 读取日志文件内容
    log_content = read_log_file(log_file_path)

    # 使用 HTML/CSS 实现上下左右滑动
    st.markdown(
        f"""
        <div style="overflow-x: auto; overflow-y: auto; white-space: pre; height: 500px; background-color: inherit;">
            {log_content}
        </div>
        """,
        unsafe_allow_html=True,
    )
    for page_num in [5, 6, 7, 8]:
        page_marker = f"###page{page_num}###"
        # if page_marker in content and page_num not in st.session_state.triggered_pages:
        if page_marker in log_content:
            
            print("跳到第五页！")
            # st.session_state.triggered_pages.add(page_num)  # 记录该页面已触发跳转
            st.switch_page(f"pages/gen_visualization_ui.py")


    # 每 0.5 秒自动刷新
    time.sleep(0.5)
    st.rerun()
main()
# st.title("Benchmark Results")

# st.markdown("---")

# st.header("The evaluation progress of the benchmark")


# progress_value = 0
# progress_bar = st.progress(progress_value)
# status_text = st.empty()

# # 模拟进度更新
# for i in range(101):
#     time.sleep(0.02)
#     progress_bar.progress(i)
#     status_text.text(f"Current Progress: {i}%")

# # 任务完成提示
# st.success("The benchmark evaluation is complete！")

# # 添加一些空白空间，使进度条区域更大
# st.markdown("<br>", unsafe_allow_html=True)  # 使用 HTML 换行增加空间

# # 下半部分：多列两行的空表格
# st.header("Result")

# columns = [
#     "GSM-8K",
#     "HumanEval",
#     "MBPP",
#     "HumanEval-CN",
#     "MBPP-CN",
#     "MMLU",
#     "GPQA",
#     "CMMLU",
#     "C-Eval",
# ]

# data = {col: [""] for col in columns}
# df = pd.DataFrame(data)

# df.index = ["score"]

# st.dataframe(df, height=50)  # 设置表格高度

# # 添加一些样式美化界面
# st.markdown(
#     """
# <style>
#     /* 调整进度条的高度和颜色 */
#     .stProgress > div > div > div {
#         height: 25px;
#         background-color: #4CAF50; /* 绿色进度条 */
#     }
#     /* 调整表格的边框和字体 */
#     .stDataFrame {
#         border: 1px solid #ccc;
#         border-radius: 5px;
#         font-size: 16px;
#     }
#     /* 调整任务完成提示的样式 */
#     .stSuccess {
#         font-size: 18px;
#         color: #4CAF50;
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )

if st.session_state.selected_button == "data_demo":
    st.switch_page("pages/gen_visualization_ui.py")
elif st.session_state.selected_button == "logs":
    st.switch_page("pages/res_visualization_ui.py")
elif st.session_state.selected_button == "training":
    st.switch_page("pages/train_visualization_ui.py")
elif st.session_state.selected_button == "benchmark":
    pass
