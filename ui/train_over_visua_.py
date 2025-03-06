import streamlit as st
import re
import pandas as pd
import altair as alt
##这是当训练跑完之后的的结果的可视化


# 读取日志文件并提取 loss 和 grad_norm
def read_log_file(log_file_path):
    steps = []
    epochs = []
    losses = []
    grad_norms = []
    
    step_counter = 0  # 用于记录 step
    epoch_markers = {}  # 记录 epoch 最后一次等于 1、2、3 时的 step
    with open(log_file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取 loss 和 grad_norm
            loss_match = re.search(r"'loss': ([\d.]+)", line)
            grad_norm_match = re.search(r"'grad_norm': ([\d.]+)", line)
            epoch_match = re.search(r"'epoch': ([\d.]+)", line)
            
            if loss_match and grad_norm_match and epoch_match:
                loss = float(loss_match.group(1))
                grad_norm = float(grad_norm_match.group(1))
                epoch = float(epoch_match.group(1))
                
                # 将数据保存到列表
                steps.append(step_counter)
                epochs.append(epoch)
                losses.append(loss)
                grad_norms.append(grad_norm)
                
                # 记录 epoch 最后一次等于 1、2、3 时的 step
                if epoch in [1, 2, 3]:
                    epoch_markers[epoch] = step_counter
                
                step_counter += 1  # 增加 step
    
    return steps, epochs, losses, grad_norms, epoch_markers

# 过滤数据，只保留 epoch 在 0-3 范围内的记录
def filter_data(steps, epochs, losses, grad_norms):
    filtered_steps = []
    filtered_epochs = []
    filtered_losses = []
    filtered_grad_norms = []
    
    for step, epoch, loss, grad_norm in zip(steps, epochs, losses, grad_norms):
        if 0 <= epoch <= 3:
            filtered_steps.append(step)
            filtered_epochs.append(epoch)
            filtered_losses.append(loss)
            filtered_grad_norms.append(grad_norm)
    
    return filtered_steps, filtered_epochs, filtered_losses, filtered_grad_norms

# 使用 Streamlit 绘制 loss 和 grad_norm 曲线
def plot_curves(data, epoch_markers):
    # 显示当前 loss 和上次 loss，用方框圈起来
    st.write("### 📊 Current and Previous Metrics")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<div style='border: 2px solid #4A90E2; padding: 15px; border-radius: 8px; background-color: #f0f2f6;'>"
                f"<h3 style='color: #4A90E2; font-size: 20px;'>Loss</h3>"
                f"<p style='font-size: 18px; color: #333;'>"
                f"<b>Current Loss</b>: {data['Loss'].iloc[-1]:.4f}<br>"
                f"<b>Previous Loss</b>: {data['Loss'].iloc[-2]:.4f}"
                f"</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div style='border: 2px solid #F5A623; padding: 15px; border-radius: 8px; background-color: #f0f2f6;'>"
                f"<h3 style='color: #F5A623; font-size: 20px;'>Gradient Norm</h3>"
                f"<p style='font-size: 18px; color: #333;'>"
                f"<b>Current Gradient Norm</b>: {data['Gradient Norm'].iloc[-1]:.4f}<br>"
                f"<b>Previous Gradient Norm</b>: {data['Gradient Norm'].iloc[-2]:.4f}"
                f"</p>"
                f"</div>",
                unsafe_allow_html=True
            )

    # 用户输入 Step 范围
    st.write("### 🔍 Select Step Range")
    min_step = int(data['Step'].min())
    max_step = int(data['Step'].max())

    # 初始化 session_state
    if "step_range" not in st.session_state:
        st.session_state.step_range = (min_step, max_step)

    # 滑动条
    step_range_slider = st.slider(
        "Step Range (Slider)",
        min_value=min_step,
        max_value=max_step,
        value=st.session_state.step_range,
        step=1,
        key="slider"
    )

    # 输入框
    col1, col2 = st.columns(2)
    with col1:
        start_step_input = st.number_input("Start Step", min_value=min_step, max_value=max_step, value=st.session_state.step_range[0], key="start_input")
    with col2:
        end_step_input = st.number_input("End Step", min_value=start_step_input, max_value=max_step, value=st.session_state.step_range[1], key="end_input")

    # 更新 session_state 时，优先使用输入框的值
    if (start_step_input, end_step_input) != st.session_state.step_range:
        st.session_state.step_range = (start_step_input, end_step_input)
    elif step_range_slider != st.session_state.step_range:
        st.session_state.step_range = step_range_slider

    # 过滤数据
    filtered_data = data[(data['Step'] >= st.session_state.step_range[0]) & (data['Step'] <= st.session_state.step_range[1])]

    # 创建两列布局
    col1, col2 = st.columns(2)

    # 绘制 loss 曲线
    with col1:
        st.write("#### 📉 Training Loss Curve")
        
        # 创建图表
        loss_chart = alt.Chart(filtered_data).mark_line(color='#1f77b4').encode(
            x=alt.X('Step', title='Step'),
            y=alt.Y('Loss', title='Loss'),
            tooltip=['Step', 'Loss']
        ).properties(
            width=400,  # 固定宽度
            height=300  # 固定高度
        )

        # 添加 epoch 虚线标记
        epoch_rules_data = pd.DataFrame({
            'Step': [epoch_markers.get(1), epoch_markers.get(2), epoch_markers.get(3)],
            'Epoch': ['epoch：1', 'epoch：2', 'epoch：3']  # 自定义显示内容
        })
        epoch_rules = alt.Chart(epoch_rules_data).mark_rule(
            color='red', strokeDash=[5, 5]
        ).encode(
            x='Step:Q',
            tooltip='Epoch:N'  # 显示自定义内容
        )

        # 启用 x 轴缩放，禁用 y 轴缩放
        zoom = alt.selection_interval(bind='scales', encodings=['x'])  # 只允许 x 轴缩放
        layered_chart = alt.layer(loss_chart, epoch_rules).add_params(zoom)
        st.altair_chart(layered_chart)

    # 绘制 grad_norm 曲线
    with col2:
        st.write("#### 📈 Gradient Norm Curve")
        
        # 创建图表
        grad_chart = alt.Chart(filtered_data).mark_line(color='#ff7f0e').encode(
            x=alt.X('Step', title='Step'),
            y=alt.Y('Gradient Norm', title='Gradient Norm'),
            tooltip=['Step', 'Gradient Norm']
        ).properties(
            width=400,  # 固定宽度
            height=300  # 固定高度
        )

        # 添加 epoch 虚线标记
        epoch_rules_data = pd.DataFrame({
            'Step': [epoch_markers.get(1), epoch_markers.get(2), epoch_markers.get(3)],
            'Epoch': ['epoch：1', 'epoch：2', 'epoch：3']  # 自定义显示内容
        })
        epoch_rules = alt.Chart(epoch_rules_data).mark_rule(
            color='red', strokeDash=[5, 5]
        ).encode(
            x='Step:Q',
            tooltip='Epoch:N'  # 显示自定义内容
        )

        # 启用 x 轴缩放，禁用 y 轴缩放
        zoom = alt.selection_interval(bind='scales', encodings=['x'])  # 只允许 x 轴缩放
        layered_chart = alt.layer(grad_chart, epoch_rules).add_params(zoom)
        st.altair_chart(layered_chart)
        
# Streamlit 应用
def main():
    st.set_page_config(layout="wide", page_title="Training Log Viewer", page_icon="📊")  # 设置宽屏布局和页面标题
    st.title("📊 Training Log Viewer")

    log_file_path = "/home/maoyingzhi2024/streamlt/log/output.log"  # 日志文件路径
    
    # 读取日志文件
    steps, epochs, losses, grad_norms, epoch_markers = read_log_file(log_file_path)
    
    if steps:
        # 过滤数据，只保留 epoch 在 0-3 范围内的记录
        steps, epochs, losses, grad_norms = filter_data(steps, epochs, losses, grad_norms)
        
        # 将数据转换为 DataFrame
        data = pd.DataFrame({
            "Step": steps,
            "Epoch": epochs,
            "Loss": losses,
            "Gradient Norm": grad_norms
        })

        # 绘制曲线
        plot_curves(data, epoch_markers)
    else:
        st.write("No training log data found.")

if __name__ == "__main__":
    main()