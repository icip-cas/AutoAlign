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
    st.switch_page("pages/page6.py")
elif st.session_state.selected_button == "training":
    pass
elif st.session_state.selected_button == "benchmark":
    st.switch_page("pages/page8.py")


# 使用缓存读取日志文件
@st.cache_data(ttl=10)  # 缓存10秒
def read_log_file(log_file_path):
    steps = []
    epochs = []
    losses = []
    grad_norms = []
    progress = []

    step_counter = 0  # 用于记录 step
    epoch_markers = {}  # 记录整数 epoch 最后一次出现时的 step
    with open(log_file_path, "r") as file:
        for line in file:
            # 使用正则表达式提取 loss 和 grad_norm
            loss_match = re.search(r"'loss': ([\d.]+)", line)
            grad_norm_match = re.search(r"'grad_norm': ([\d.]+)", line)
            epoch_match = re.search(r"'epoch': ([\d.]+)", line)
            progress_match = re.search(r"(\d+)%", line)

            if loss_match and grad_norm_match and epoch_match:
                loss = float(loss_match.group(1))
                grad_norm = float(grad_norm_match.group(1))
                epoch = float(epoch_match.group(1))

                # 将数据保存到列表
                steps.append(step_counter)
                epochs.append(epoch)
                losses.append(loss)
                grad_norms.append(grad_norm)

                # 如果 epoch 是整数，记录其最后一次出现时的 step
                if epoch.is_integer():
                    epoch_markers[int(epoch)] = step_counter

                step_counter += 1  # 增加 step

            if progress_match:
                progress.append(int(progress_match.group(1)))

    return steps, epochs, losses, grad_norms, epoch_markers, progress


# 使用 Streamlit 绘制 loss 和 grad_norm 曲线
def plot_curves(data, epoch_markers, progress):
    # 显示当前 loss 和上次 loss，用方框圈起来
    st.write("### 📊 Current and Previous Metrics")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<div style='border: 2px solid #4A90E2; padding: 20px; border-radius: 15px; background-color: #f0f2f6; box-shadow: 5px 5px 15px rgba(0,0,0,0.1);'>"
                f"<h3 style='color: #4A90E2; font-size: 1.6em; margin-bottom: 15px; text-align: center; font-family: Arial, sans-serif;'>Loss</h3>"
                f"<div style='display: flex; justify-content: space-between; font-size: 1.3em; color: #333; margin-bottom: 10px; font-family: Arial, sans-serif;'>"
                f"<div><b>Previous Loss</b>: {data['Loss'].iloc[-2]:.4f}</div>"
                f"<div><b>Current Loss</b>: {data['Loss'].iloc[-1]:.4f}</div>"
                f"</div>"
                f"<div style='text-align: center; font-size: 1.5em; color: #333; font-weight: bold; font-family: Arial, sans-serif;'><b>Change</b>: {data['Loss'].iloc[-1] - data['Loss'].iloc[-2]:.4f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div style='border: 2px solid #4A90E2; padding: 20px; border-radius: 15px; background-color: #f0f2f6; box-shadow: 5px 5px 15px rgba(0,0,0,0.1);'>"
                f"<h3 style='color: #4A90E2; font-size: 1.6em; margin-bottom: 15px; text-align: center; font-family: Arial, sans-serif;'>Gradient Norm</h3>"
                f"<div style='display: flex; justify-content: space-between; font-size: 1.3em; color: #333; margin-bottom: 10px; font-family: Arial, sans-serif;'>"
                f"<div><b>Previous Gradient Norm</b>: {data['Gradient Norm'].iloc[-2]:.4f}</div>"
                f"<div><b>Current Gradient Norm</b>: {data['Gradient Norm'].iloc[-1]:.4f}</div>"
                f"</div>"
                f"<div style='text-align: center; font-size: 1.5em; color: #333; font-weight: bold; font-family: Arial, sans-serif;'><b>Change</b>: {data['Gradient Norm'].iloc[-1] - data['Gradient Norm'].iloc[-2]:.4f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    # 显示进度条
    if progress:
        st.write("### 🚀 Training Progress")
        current_progress = progress[-1]
        st.progress(current_progress / 100)
        st.write(f"**Current Progress:**​ {current_progress}%")

    # 用户输入 Step 范围
    st.write("### 🔍 Select Step Range")
    min_step = int(data["Step"].min())
    max_step = int(data["Step"].max())

    # 初始化 session_state
    if "step_range" not in st.session_state:
        st.session_state.step_range = (min_step, max_step)
    if "user_selected" not in st.session_state:
        st.session_state.user_selected = False
    if "start_input" not in st.session_state:
        st.session_state.start_input = min_step
    if "end_input" not in st.session_state:
        st.session_state.end_input = max_step

    # 滑动条
    def update_slider():
        st.session_state.step_range = (
            st.session_state.start_input,
            st.session_state.end_input,
        )

    step_range_slider = st.slider(
        "Step Range (Slider)",
        min_value=min_step,
        max_value=max_step,
        value=st.session_state.step_range,
        step=1,
        key="step_slider",
        on_change=update_slider,
    )
    # 使用 CSS 将按钮居中
    st.markdown(
        """
        <style>
        .centered {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 将按钮放入一个居中的容器
    with st.container():
        st.markdown('<div class="centered">', unsafe_allow_html=True)
        if st.button("↻"):
            st.session_state.user_selected = False
            st.session_state.step_range = (min_step, max_step)
            st.session_state.start_input = min_step
            st.session_state.end_input = max_step
        st.markdown("</div>", unsafe_allow_html=True)

    # 输入框
    col1, col2 = st.columns(2)
    with col1:
        start_step_input = st.number_input(
            "Start Step",
            min_value=min_step,
            max_value=max_step,
            value=st.session_state.start_input,
            key="start_step_input",
            on_change=lambda: setattr(
                st.session_state,
                "step_range",
                (st.session_state.start_input, st.session_state.end_input),
            ),
        )
    with col2:
        end_step_input = st.number_input(
            "End Step",
            min_value=start_step_input,
            max_value=max_step,
            value=st.session_state.end_input,
            key="end_step_input",
            on_change=lambda: setattr(
                st.session_state,
                "step_range",
                (st.session_state.start_input, st.session_state.end_input),
            ),
        )

    # 状态更新逻辑
    if (
        start_step_input != st.session_state.start_input
        or end_step_input != st.session_state.end_input
    ):
        st.session_state.start_input = start_step_input
        st.session_state.end_input = end_step_input
        st.session_state.step_range = (start_step_input, end_step_input)
        st.session_state.user_selected = True
    elif step_range_slider != st.session_state.step_range:
        st.session_state.step_range = step_range_slider
        st.session_state.start_input = step_range_slider[0]
        st.session_state.end_input = step_range_slider[1]
        st.session_state.user_selected = True

    # 如果用户未手动选择范围，则实时更新到最新 step
    if not st.session_state.user_selected:
        st.session_state.step_range = (min_step, max_step)
        st.session_state.start_input = min_step
        st.session_state.end_input = max_step

    # 过滤数据
    filtered_data = data[
        (data["Step"] >= st.session_state.step_range[0])
        & (data["Step"] <= st.session_state.step_range[1])
    ]

    # 使用 CSS 将图表居中
    st.markdown(
        """
        <style>
        .stChart {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        # 创建图表
        loss_chart = (
            alt.Chart(filtered_data)
            .mark_line(color="#1f77b4")
            .encode(
                x=alt.X("Step", title="Step"),
                y=alt.Y("Loss", title="Loss"),
                tooltip=["Step", "Loss"],
            )
            .properties(
                width=600,  # 增加宽度
                height=400,  # 增加高度
            )
        )

        # 添加 epoch 虚线标记（过滤掉 epoch=0）
        epoch_rules_data = pd.DataFrame(
            {
                "Step": [
                    epoch_markers[epoch]
                    for epoch in sorted(epoch_markers.keys())
                    if epoch != 0
                ],
                "Epoch": [
                    f"epoch: {epoch}"
                    for epoch in sorted(epoch_markers.keys())
                    if epoch != 0
                ],
            }
        )
        epoch_rules = (
            alt.Chart(epoch_rules_data)
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x="Step:Q", tooltip="Epoch:N")
        )

        # 添加最后一个 epoch 的虚线标记
        if epoch_markers:
            last_epoch = max(epoch_markers.keys())
            last_epoch_rule = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "Step": [epoch_markers[last_epoch]],
                            "Epoch": [f"Last Epoch: {last_epoch}"],
                        }
                    )
                )
                .mark_rule(color="green", strokeDash=[5, 5])
                .encode(x="Step:Q", tooltip="Epoch:N")
            )
            epoch_rules = epoch_rules + last_epoch_rule

        # 启用 x 轴缩放，禁用 y 轴缩放
        zoom = alt.selection_interval(bind="scales", encodings=["x"])
        layered_chart = alt.layer(loss_chart, epoch_rules).add_params(zoom)

        # 添加文字到图表上方
        title_text = (
            alt.Chart(pd.DataFrame({"text": ["📉 Training Loss Curve"]}))
            .mark_text(
                align="center",
                baseline="top",
                fontSize=32,
                fontWeight="bold",
                color="white",
                dy=-20,
            )
            .encode(text="text:N")
        )

        # 组合图表和文字
        final_chart = alt.vconcat(title_text, layered_chart).properties(spacing=0)

        # 渲染图表
        st.altair_chart(final_chart, use_container_width=True)

    with col2:
        # 创建图表
        grad_chart = (
            alt.Chart(filtered_data)
            .mark_line(color="#ff7f0e")
            .encode(
                x=alt.X("Step", title="Step"),
                y=alt.Y("Gradient Norm", title="Gradient Norm"),
                tooltip=["Step", "Gradient Norm"],
            )
            .properties(
                width=600,  # 增加宽度
                height=400,  # 增加高度
            )
        )

        # 添加 epoch 虚线标记（过滤掉 epoch=0）
        epoch_rules_data = pd.DataFrame(
            {
                "Step": [
                    epoch_markers[epoch]
                    for epoch in sorted(epoch_markers.keys())
                    if epoch != 0
                ],
                "Epoch": [
                    f"epoch: {epoch}"
                    for epoch in sorted(epoch_markers.keys())
                    if epoch != 0
                ],
            }
        )
        epoch_rules = (
            alt.Chart(epoch_rules_data)
            .mark_rule(color="red", strokeDash=[5, 5])
            .encode(x="Step:Q", tooltip="Epoch:N")
        )

        # 添加最后一个 epoch 的虚线标记
        if epoch_markers:
            last_epoch = max(epoch_markers.keys())
            last_epoch_rule = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "Step": [epoch_markers[last_epoch]],
                            "Epoch": [f"Last Epoch: {last_epoch}"],
                        }
                    )
                )
                .mark_rule(color="green", strokeDash=[5, 5])
                .encode(x="Step:Q", tooltip="Epoch:N")
            )
            epoch_rules = epoch_rules + last_epoch_rule

        # 启用 x 轴缩放，禁用 y 轴缩放
        zoom = alt.selection_interval(bind="scales", encodings=["x"])
        layered_chart = alt.layer(grad_chart, epoch_rules).add_params(zoom)

        # 添加文字到图表上方
        title_text = (
            alt.Chart(pd.DataFrame({"text": ["📈 Gradient Norm Curve"]}))
            .mark_text(
                align="center",
                baseline="top",
                fontSize=32,
                fontWeight="bold",
                color="white",
                dy=-20,
            )
            .encode(text="text:N")
        )

        # 组合图表和文字
        final_chart = alt.vconcat(title_text, layered_chart).properties(spacing=0)

        # 渲染图表
        st.altair_chart(final_chart, use_container_width=True)


# Streamlit 应用
def main():
    st.title("📊 Training Log Viewer")

    log_file_path = (
        "/141nfs/wangjunxiang/AutoAlign/testing-data/output.log"  # 日志文件路径
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
    steps, epochs, losses, grad_norms, epoch_markers, progress = read_log_file(
        log_file_path
    )

    if steps:
        # 将数据转换为 DataFrame
        data = pd.DataFrame(
            {
                "Step": steps,
                "Epoch": epochs,
                "Loss": losses,
                "Gradient Norm": grad_norms,
            }
        )

        # 绘制曲线
        plot_curves(data, epoch_markers, progress)
    else:
        st.write("No training log data found.")

    # 每 5 秒自动刷新
    time.sleep(5)
    st.rerun()


main()
