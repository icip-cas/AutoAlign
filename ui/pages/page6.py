import os
import streamlit as st
import re
import pandas as pd
import altair as alt
import time
from pages.navbar import render_navbar_visual
from streamlit_autorefresh import st_autorefresh
st.set_page_config(layout="wide", page_title="Training Log Viewer", page_icon="📊")


render_navbar_visual()

if "triggered_pages" not in st.session_state:
    st.session_state["triggered_pages"] = set()

st_autorefresh(interval=2000, key="refresh")
log_path = "test.log"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        content = f.read()
    for page_num in [6, 7, 8]:
        page_marker = f"###page{page_num}###"
        if page_marker in content and page_num not in st.session_state.triggered_pages:
            st.session_state.triggered_pages.add(page_num)  # 记录该页面已触发跳转
            st.switch_page(f"page{page_num}.py")

if st.session_state.selected_button == "data_demo":
    st.switch_page("pages/page5.py")
elif st.session_state.selected_button == "logs":
    pass
elif st.session_state.selected_button == "training":
    st.switch_page("pages/page7.py")
elif st.session_state.selected_button == "benchmark":
    st.switch_page("pages/page8.py")


# 使用缓存读取日志文件
@st.cache_data(ttl=0.2)  # 缓存10秒
def read_log_file(log_file_path):
    current_generate = []
    total_generate = []
    all_total_generate = 0
    algorithm: str = None
    pid: str = None
    pid_dict = {}
    unmatched_lines = []  # 用于保存未匹配的行

    with open(log_file_path, "r") as file:
        for line in file:
            # 匹配 "single_process_inference pid=" 这一行，并提取 pid
            pid_match = re.search(r"pid=(\d+)", line)
            if pid_match:
                pid = pid_match.group(1)
                if pid not in pid_dict:
                    pid_dict[pid] = []
            # 匹配 "###algorithm###" 这一行，并提取后面的字符串
            elif line.startswith("###algorithm###"):
                match = re.search(r"###algorithm###\s*(\S+)", line)
                if match:
                    algorithm = match.group(1)
                continue
            # 匹配 "###all_data_len###" 这一行，并提取后面的数值
            elif line.startswith("###all_data_len###"):
                match = re.search(r"###all_data_len###\s*(\d+)", line)
                if match:
                    all_total_generate = int(match.group(1))
                continue
            # 匹配 "Processed prompts:" 这一行
            elif line.startswith("Processed prompts:"):
                match = re.search(r"(\d+)/(\d+)", line)
                if match:
                    num1 = int(match.group(1))
                    num2 = int(match.group(2))
                if (num1, num2) not in pid_dict[pid]:
                    pid_dict[pid].append((num1, num2))
                    current_generate.append(num1)
                    total_generate.append(num2)
            else:
                unmatched_lines.append(line)  # 将未匹配的行添加到 unmatched_lines

    return (
        current_generate,
        total_generate,
        algorithm,
        all_total_generate,
        pid_dict,
        unmatched_lines,
    )


# 使用 Streamlit 绘制 loss 和 grad_norm 曲线
def plot_curves(pid_dict, all_total_generate):
    # 显示进度条
    current_generate_sum = sum(
        pid_dict[pid][-1][0] if pid_dict.get(pid) else 0 for pid in pid_dict
    )
    if pid_dict:
        st.write("### 🚀 Training Progress")
        st.progress(round(current_generate_sum / all_total_generate, 2))
        st.write(
            f"**Current Progress:**​ {round(current_generate_sum / all_total_generate, 2) * 100}%"
        )


# Streamlit 应用
def main():
    st.title("📊 Training Log Viewer")

    log_file_path = (
        "/141nfs/wangjunxiang/AutoAlign/testing-data/txt.log"  # 日志文件路径
    )

    # 初始化 session_state
    if "step_range" not in st.session_state:
        st.session_state.step_range = (0, 0)
    if "user_selected" not in st.session_state:
        st.session_state.user_selected = False
    if "start_input" not in st.session_state:
        st.session_state.start_input = 0
    if "end_input" not in st.session_state:
        st.session_state.end_input = 0

    # 读取日志文件
    (
        current_generate,
        total_generate,
        algorithm,
        all_total_generate,
        pid_dict,
        unmatched_lines,
    ) = read_log_file(log_file_path)
    for pid in pid_dict:
        pid_dict[pid].sort(key=lambda x: x[0])
    # 不同方法的all_total_generate不一样
    if algorithm == "CAI":
        all_total_generate = all_total_generate * 3
        all_total_generate = max(
            all_total_generate,
            sum(pid_dict[pid][-1][0] if pid_dict.get(pid) else 0 for pid in pid_dict),
        )
    # 绘制曲线
    plot_curves(pid_dict, all_total_generate)

    # 将 unmatched_lines 以换行符连接成字符串
    log_content = "\n".join(unmatched_lines)

    # st.write("Log:")
    # st.markdown("<h2 style='font-size: 24px;'>Log:</h2>", unsafe_allow_html=True)
    st.text_area(label="", value=log_content, height=500)

    # 每 5 秒自动刷新
    time.sleep(0.2)
    st.rerun()


# if __name__ == "__main__":
#     st.write("bruh")
#     main()

main()
