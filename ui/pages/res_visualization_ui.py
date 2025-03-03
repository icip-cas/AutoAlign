import os
import streamlit as st
import re
from pages.navigation import render_navbar_visual
from streamlit_autorefresh import st_autorefresh

# 页面基础设置
st.set_page_config(layout="wide", page_title="Generate Answer", page_icon="📊")
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 渲染导航栏
render_navbar_visual()

# 页面跳转检查
if "triggered_pages" not in st.session_state:
    st.session_state["triggered_pages"] = set()

st_autorefresh(interval=2000, key="refresh")  # 每2秒自动刷新

# 页面跳转逻辑（保留原有功能）
log_path = "outputs/self_rewarding_log.log"
if os.path.exists(log_path):
    with open(log_path, "r") as f:
        content = f.read()
    for page_num in [5, 6, 7, 8]:
        if f"###page{page_num}###" in content:
            if page_num == 5:
                st.switch_page(f"pages/gen_visualization_ui.py")
            if page_num == 6:
                st.switch_page(f"pages/res_visualization_ui.py")
            if page_num == 7:
                st.switch_page(f"pages/train_visualization_ui.py")
            if page_num == 8:
                st.switch_page(f"pages/eval_visualization_ui.py")

# 导航栏页面切换（保留原有功能）
if st.session_state.selected_button == "data_demo":
    st.switch_page("pages/gen_visualization_ui.py")
elif st.session_state.selected_button == "training":
    st.switch_page("pages/train_visualization_ui.py")
elif st.session_state.selected_button == "benchmark":
    st.switch_page("pages/eval_visualization_ui.py")

# 主要日志显示功能
st.title("📊 Generate Answer")
log_file_path = "outputs/self_rewarding_log.log"

try:
    with open(log_file_path, "r") as file:
        log_content = file.read()
    st.text_area(label="", value=log_content, height=700)
except FileNotFoundError:
    st.warning("Log file not found, waiting for generation to start...")
except Exception as e:
    st.error(f"Error reading log file: {str(e)}")