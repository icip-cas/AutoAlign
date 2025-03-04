import streamlit as st

# 初始化会话状态
if 'selected_button' not in st.session_state:
    st.session_state.selected_button = "data_gen"  # 默认选中数据生成

# 动态导航栏
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("📥 数据生成", use_container_width=True):
        st.session_state.selected_button = "data_gen"
with nav_cols[1]:
    if st.button("🔎 数据筛选", use_container_width=True):
        st.session_state.selected_button = "data_filter"
with nav_cols[2]:
    if st.button("🎓 训练", use_container_width=True):
        st.session_state.selected_button = "train"
with nav_cols[3]:
    if st.button("📊 评测", use_container_width=True):
        st.session_state.selected_button = "eval"

# 动态显示对应内容
if st.session_state.selected_button == "data_gen":
    st.title("Data Synthesis")
    
    with st.form("synthesis_form"):
        # 合成方法选择
        method = st.selectbox("Synthesis Method", 
                            ["MAGPIE", "Self-Instruct", "Back-Translation"])
        
        # 动态参数区域
        params = {}
        if method == "MAGPIE":
            cols = st.columns(3)
            with cols[0]:
                params["temperature"] = st.number_input("Temperature", 0.0, 1.0, 0.7)
            with cols[1]:
                params["top_p"] = st.number_input("Top-p", 0.0, 1.0, 0.9)
            with cols[2]:
                params["max_length"] = st.number_input("Max Length", 50, 500, 100)
        
        elif method == "Self-Instruct":
            cols = st.columns([3,2])
            with cols[0]:
                params["num_instructions"] = st.number_input("指令数量", 1, 1000, 100)
            with cols[1]:
                params["batch_size"] = st.selectbox("批大小", [16, 32, 64])
        
        elif method == "Back-Translation":
            cols = st.columns(2)
            with cols[0]:
                params["src_lang"] = st.selectbox("源语言", ["en", "zh", "es"])
            with cols[1]:
                params["tgt_lang"] = st.selectbox("目标语言", ["en", "zh", "es"])
        
        # 路径配置
        st.subheader("路径配置")
        input_dir = st.text_input("输入目录", placeholder="输入原始数据路径")
        output_dir = st.text_input("输出目录", placeholder="指定结果保存路径")
        
        # 提交按钮
        if st.form_submit_button("🚀 开始生成"):
            if not all([input_dir, output_dir]):
                st.error("请填写完整路径信息")
            else:
                st.session_state.generation_params = {
                    "method": method,
                    "params": params,
                    "paths": {"input": input_dir, "output": output_dir}
                }
                st.success("参数保存成功！")

elif st.session_state.selected_button == "data_filter":
    st.title("Data Filtering")
    # 添加数据筛选相关控件...

elif st.session_state.selected_button == "train":
    st.title("Model Training")
    # 添加训练相关控件...

elif st.session_state.selected_button == "eval":
    st.title("Model Evaluation")
    # 添加评测相关控件...