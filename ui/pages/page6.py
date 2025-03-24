import os
import streamlit as st
import re
import pandas as pd
import altair as alt
import time
from pages.navbar import render_navbar_visual
from streamlit_autorefresh import st_autorefresh
st.set_page_config(layout="wide", page_title="Training Log Viewer", page_icon="📊")
hide_sidebar_css = """
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_sidebar_css, unsafe_allow_html=True)

render_navbar_visual()

if "triggered_pages" not in st.session_state:
    st.session_state["triggered_pages"] = set()

st_autorefresh(interval=2000, key="refresh")
log_path = "outputs/self_rewarding_log.log"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        content = f.read()
    for page_num in [5, 6, 7, 8]:
        page_marker = f"###page{page_num}###"
        # if page_marker in content and page_num not in st.session_state.triggered_pages:
        if page_marker in content:
            # 
            print("跳到第七页！")
            # st.session_state.triggered_pages.add(page_num)  # 记录该页面已触发跳转
            st.switch_page(f"pages/page{page_num}.py")
    

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
    progress_dict = {}
    unmatched_lines = []
    data_length = None  # 新增变量存储数据长度

    with open(log_file_path, "r") as file:
        for line in file: 
            # 新增：匹配Data Length行 (不区分大小写，允许冒号后空格)
            data_match = re.search(r'Data Length:\s*(\d+)', line, re.IGNORECASE)
            if data_match:
                data_length = int(data_match.group(1))
                continue  # 匹配成功后跳过后续判断
                
            # 匹配进度条行
            progress_match = re.search(r'(\d+)/(\d+)\s+\[', line)
            if progress_match:
                numerator = int(progress_match.group(1))
                denominator = int(progress_match.group(2))
                if denominator in progress_dict:
                    progress_dict[denominator] = max(progress_dict[denominator], numerator)
                else:
                    progress_dict[denominator] = numerator
            else:
                unmatched_lines.append(line)

    return progress_dict, unmatched_lines, data_length  # 返回新增参数

def plot_curves(progress_dict, all_total_generate):
    if all_total_generate == 0:
        return
        
    current_generate_sum = sum(progress_dict.values())
    progress = current_generate_sum / all_total_generate
    st.write("### 🚀 Training Progress")
    st.progress(min(progress, 1.0))
    st.write(f"**Current Progress:** {progress * 100:.2f}%")

def main():
    st.title("📊 Training Log Viewer")
    log_file_path = "outputs/self_rewarding_log.log"

    # 读取日志文件（接收新增的data_length参数）
    progress_dict, unmatched_lines, data_length = read_log_file(log_file_path)
    
    # 处理特殊算法逻辑（示例保持原有逻辑，可根据需要修改）
    all_total_generate = data_length * 5


    # 显示进度（保持不变）
    plot_curves(progress_dict, all_total_generate)

    # 显示日志内容（保持不变）
    log_content = "\n".join(unmatched_lines)
    st.text_area(label="", value=log_content, height=500)

    # 自动刷新（保持不变）
    time.sleep(0.2)
    st.rerun()


# if __name__ == "__main__":
#     st.write("bruh")
#     main()

main()
