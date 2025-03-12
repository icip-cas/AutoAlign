import streamlit as st
import time

# 使用缓存读取日志文件，缓存 0.5 秒
@st.cache_data(ttl=0.5)
def read_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    return log_content

# Streamlit 应用
def main():
    st.set_page_config(layout="wide", page_title="Real-Time Log Viewer", page_icon="📄")
    st.title("📄 Eval Log Viewer")

    log_file_path = "/home/maoyingzhi2024/streamlt/log/generate_log.log"  # 日志文件路径

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

    # 每 0.5 秒自动刷新
    time.sleep(0.5)
    st.rerun()

if __name__ == "__main__":
    main()