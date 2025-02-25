import os
import streamlit as st
from transformers import AutoTokenizer

# 隐藏侧边栏
hide_sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# 初始化 session_state 变量
for key, default in {
    "mod_dir": '',
    "file_dir": '',
    "submit_clicked": False,
    "show_turn": False,
    "show_source": False,
    "show_token": False,
    "show_time_dis": False,
    "chat_template": ''
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 表单输入
with st.form('input_dir'):
    st.session_state.mod_dir = st.text_input('Model Directory:')
    st.session_state.file_dir = st.text_input('File Directory')
    st.session_state.show_turn = st.checkbox('Show Turn Distribution')
    st.session_state.show_source = st.checkbox('Show Source Distribution')
    st.session_state.show_token = st.checkbox('Show Token Distribution')
    st.session_state.show_time_dis = st.checkbox('Show Time Distribution')
    st.session_state.chat_template = st.selectbox(
        'Template Select',
        ("vicuna_v1.1", "llama-2-chat", "llama-2-chat-keep-system", "chatml",
         "chatml-keep-system", "llama-3-instruct", "mistral-instruct", 
         "gemma", "zephyr", "chatml-idsys", "glm-4-chat", 
         "glm-4-chat-keep-system", "default")
    )
    st.session_state.data_type = st.selectbox(
        'Type Select',
        (
            "sft", "dpo"
        )
    )
    submit = st.form_submit_button('Submit')
    if submit:
        st.session_state.submit_clicked = True

# 获取用户输入的目录
mod_dir = str(st.session_state.mod_dir)
file_dir = str(st.session_state.file_dir)

# 验证模型和文件路径
if st.session_state.submit_clicked:
    mod_available = True
    path_available = os.path.exists(file_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(mod_dir)
    except Exception:
        mod_available = False
        st.write('Invalid model directory!')

    if not path_available:
        st.write('Invalid data directory!')

    if mod_available and path_available:
        st.switch_page("page2.py")
