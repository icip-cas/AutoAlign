import streamlit as st

# 页面标题
st.title("Data Synthesis")

# 数据生成方法选择
method = st.selectbox("Synthesis Method", ["-", "MAGPIE", "Self-Instruct", "Back-Translation"])

# 根据选择的方法动态显示参数
if method != "-":
    st.subheader(f"{method} Parameter Configuration")
    
    # MAGPIE 方法参数
    if method == "MAGPIE":
        # 基础配置
        st.subheader("基础配置")
        cols = st.columns(2)
        with cols[0]:
            task_name = st.text_input("任务名称", placeholder="请输入任务名称")
        with cols[1]:
            timestamp = st.text_input("时间戳", placeholder="自动生成或手动输入")

        # 路径配置
        st.subheader("路径配置")
        cols = st.columns(2)
        with cols[0]:
            model_path = st.text_input("模型路径", placeholder="请输入模型路径")
        with cols[1]:
            config_path = st.text_input("Config 路径", placeholder="请输入配置文件路径")

        # 模型相关配置
        st.subheader("模型配置")

        model_id = st.text_input("模型 ID", placeholder="与 Huggingface 路径对应")
        
        cols = st.columns(2)
        with cols[0]:
            tensor_parallel = st.number_input("Tensor Parallel", min_value=1, value=1)
        with cols[1]:
            gpu_utilization = st.slider("GPU 最大利用率", 0.0, 1.0, 0.8)

        # 采样配置
        st.subheader("采样配置")
        cols = st.columns(3)
        with cols[0]:
            total_prompts = st.number_input("Total Prompts", min_value=1, value=100)
        with cols[1]:
            temperature = st.number_input("温度", min_value=0.0, max_value=2.0, value=0.7)
        with cols[2]:
            top_p = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=0.9)

        # 设备列表
        st.subheader("设备列表")
        devices = st.multiselect(
            "选择设备",
            options=["GPU 0", "GPU 1", "GPU 2", "GPU 3"],
            default=["GPU 0"]
        )

    
    # Self-Instruct 方法参数
    elif method == "Self-Instruct":
        cols = st.columns([3, 2])
        with cols[0]:
            num_instructions = st.number_input("指令数量", 1, 1000, 100)
        with cols[1]:
            batch_size = st.selectbox("批大小", [16, 32, 64])
    
    # Back-Translation 方法参数
    elif method == "Back-Translation":
        cols = st.columns(2)
        with cols[0]:
            src_lang = st.selectbox("源语言", ["en", "zh", "es"])
        with cols[1]:
            tgt_lang = st.selectbox("目标语言", ["en", "zh", "es"])
        back_translate_rounds = st.number_input("翻译轮次", 1, 5, 2)
    
    # 路径配置
    st.subheader("路径配置")
    input_dir = st.text_input("输入目录", placeholder="输入原始数据路径")
    output_dir = st.text_input("输出目录", placeholder="指定结果保存路径")
    
    # 提交按钮
    if st.button("🚀 开始生成"):
        if not all([input_dir, output_dir]):
            st.error("请填写完整路径信息")
        else:
            st.success("参数配置成功！")
            # 将参数保存到 session_state 或传递给后端
            st.session_state.generation_params = {
                "method": method,
                "params": locals()  # 保存所有局部变量
            }