import streamlit as st
import time
hide_sidebar_css = """
<style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_sidebar_css, unsafe_allow_html=True)
# 设置页面标题
st.title("Loading ...")

# 使用 Markdown 美化加载信息
st.markdown(
    """
    <style>
    .loading-text {
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-top: 20px;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .progress-percent {
        font-size: 18px;
        font-weight: bold;
        color: #2575fc;
        text-align: center;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 美化加载信息
st.markdown(
    '<p class="loading-text">Please wait while we process your request...</p>',
    unsafe_allow_html=True,
)

# 创建一个进度条
progress_bar = st.progress(0)

# 创建一个文本组件用于显示百分比
progress_percent = st.empty()

# 模拟加载过程
for i in range(100):
    # 每次循环增加1%的进度
    progress_bar.progress(i + 1)

    # 更新百分比文本
    progress_percent.markdown(
        f'<p class="progress-percent">{i + 1}%</p>',
        unsafe_allow_html=True,
    )

    # 模拟加载时间，每次循环暂停0.5秒
    time.sleep(0.5)

# 加载完成后显示完成信息
st.markdown(
    '<p class="loading-text">Processing complete! Redirecting...</p>',
    unsafe_allow_html=True,
)

# 跳转到其他页面
time.sleep(2)  # 给用户一点时间看到成功信息
st.switch_page("pages/page5.py")
