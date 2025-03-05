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
        with st.form("magpie_config_form"):
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

            # 提交按钮
            cols1,cols2, cols3 = st.columns([5,3,4])
            with cols2:
                submitted = st.form_submit_button("🚀 提交配置")

            # 提交后验证和处理
            if submitted:
                # 检查所有必填字段
                missing_fields = []
                if not task_name:
                    missing_fields.append("任务名称")
                if not timestamp:
                    missing_fields.append("时间戳")
                if not model_path:
                    missing_fields.append("模型路径")
                if not config_path:
                    missing_fields.append("Config 路径")
                if not model_id:
                    missing_fields.append("模型 ID")
                if not tensor_parallel:
                    missing_fields.append("Tensor Parallel")
                if not gpu_utilization:
                    missing_fields.append("GPU 最大利用率")
                if not total_prompts:
                    missing_fields.append("Total Prompts")
                if not temperature:
                    missing_fields.append("温度")
                if not top_p:
                    missing_fields.append("Top-p")
                if not devices:
                    missing_fields.append("设备列表")

                # 如果有未填字段，提示具体哪个字段未填
                if missing_fields:
                    st.error(f"以下字段未填写：{', '.join(missing_fields)}")
                else:
                    # 保存配置到 session_state
                    st.session_state.magpie_config = {
                        "task_name": task_name,
                        "timestamp": timestamp,
                        "model_path": model_path,
                        "config_path": config_path,
                        "model_id": model_id,
                        "tensor_parallel": tensor_parallel,
                        "gpu_utilization": gpu_utilization,
                        "total_prompts": total_prompts,
                        "temperature": temperature,
                        "top_p": top_p,
                        "devices": devices
                    }
                    st.success("配置保存成功！")
                    # 跳转到 page3.py
                    st.switch_page("page3.py")

    
    # Self-Instruct 方法参数
    elif method == "Self-Instruct":

        with st.form("self_instruct_config_form"):
            # 基础配置
            st.subheader("基础配置")
            cols = st.columns([3, 2])
            with cols[0]:
                model_id = st.text_input("模型 ID", placeholder="请输入模型名称（可自定义）")
            with cols[1]:
                template_name = st.text_input("模板名称", placeholder="请输入 ATA 规定的模板名称")

            # 路径配置
            st.subheader("路径配置")
            cols = st.columns(2)
            with cols[0]:
                question_gen_model_path = st.text_input("Self-Instruct 模型路径", placeholder="请输入模型路径")
            with cols[1]:
                seed_data_path = st.text_input("Seed-data-path", placeholder="请输入 ShareGPT 格式的种子数据路径")

            # 推理配置
            st.subheader("推理配置")
            cols = st.columns(2)
            with cols[0]:
                backend = st.selectbox("推理后端", ["hf", "vllm"])
            with cols[1]:
                num_prompts = st.number_input("Self-Instruct 次数", min_value=1, value=10)

            # 提交按钮
            cols1,cols2, cols3 = st.columns([5,3,4])
            with cols2:
                submitted = st.form_submit_button("🚀 提交配置")

            # 提交后验证和处理
            if submitted:
                # 检查所有必填字段
                missing_fields = []
                if not model_id:
                    missing_fields.append("模型 ID")
                if not template_name:
                    missing_fields.append("模板名称")
                if not question_gen_model_path:
                    missing_fields.append("Self-Instruct 模型路径")
                if not seed_data_path:
                    missing_fields.append("seed-data-path")
                if not backend:
                    missing_fields.append("推理后端")
                if not num_prompts:
                    missing_fields.append("Self-Instruct 次数")

                # 如果有未填字段，提示具体哪个字段未填
                if missing_fields:
                    st.error(f"以下字段未填写：{', '.join(missing_fields)}")
                else:
                    # 保存配置到 session_state
                    st.session_state.self_instruct_config = {
                        "model_id": model_id,
                        "template_name": template_name,
                        "question_gen_model_path": question_gen_model_path,
                        "seed_data_path": seed_data_path,
                        "backend": backend,
                        "num_prompts": num_prompts,
                        "num_instructions": num_instructions,
                        "batch_size": batch_size
                    }
                    st.success("配置保存成功！")
                    # 跳转到 page3.py
                    st.switch_page("page3.py")
                                #这里还需要加一个跳转！！！！！！！！！"""
            
    # Back-Translation 方法参数
    elif method == "Back-Translation":
        with st.form("back_translation_config_form"):

            # 采样配置
            st.subheader("采样配置")
            cols = st.columns(3)
            with cols[0]:
                temperature = st.number_input("温度", min_value=0.0, max_value=2.0, value=0.7)
            with cols[1]:
                top_p = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=0.9)
            with cols[2]:
                max_length = st.number_input("最长长度", min_value=50, max_value=1000, value=512)

            # 路径配置
            st.subheader("路径配置")
            cols = st.columns(2)
            with cols[0]:
                unlabeled_data_path = st.text_input("无标注数据路径", placeholder="请输入无标注数据路径")
            with cols[1]:
                output_path = st.text_input("生成文件保存路径", placeholder="请输入生成文件保存路径")
            back_translation_model_path = st.text_input("回译模型路径", placeholder="请输入回译模型路径")

            st.subheader("推理配置")
            tensor_parallel_size = st.number_input("Tensor Parallel Size", min_value=1, max_value=8, value=1, help="每个模型占用几张卡推理")

            # 提交按钮
            cols1,cols2, cols3 = st.columns([5,3,4])
            with cols2:
                submitted = st.form_submit_button("🚀 提交配置")

            # 提交后验证和处理
            if submitted:
                # 检查所有必填字段
                missing_fields = []
                if not temperature:
                    missing_fields.append("温度")
                if not top_p:
                    missing_fields.append("Top-p")
                if not max_length:
                    missing_fields.append("最长长度")
                if not unlabeled_data_path:
                    missing_fields.append("无标注数据路径")
                if not output_path:
                    missing_fields.append("生成文件保存路径")
                if not back_translation_model_path:
                    missing_fields.append("回译模型路径")
                if not tensor_parallel_size:
                    missing_fields.append("Tensor Parallel Size")

                # 如果有未填字段，提示具体哪个字段未填
                if missing_fields:
                    st.error(f"以下字段未填写：{', '.join(missing_fields)}")
                else:
                    # 保存配置到 session_state
                    st.session_state.back_translation_config = {
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_length": max_length,
                        "unlabeled_data_path": unlabeled_data_path,
                        "output_path": output_path,
                        "back_translation_model_path": back_translation_model_path,
                        "tensor_parallel_size": tensor_parallel_size
                    }
                    st.success("配置保存成功！")
                    # 跳转到 page3.py
                    st.switch_page("page3.py")