import streamlit as st
import time
import os
st.set_page_config(layout="centered")
# 初始化会话状态

for key, default in {
    "p1_fin": False,
    "p2_fin": False,
    "p3_fin": False,
    "p4_fin": False,
    "selected_button": "data_gen"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# 动态导航栏
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("📥 Generate Query", use_container_width=True):
        st.session_state.selected_button = "data_gen"
with nav_cols[1]:
    if st.button("🔎 Sample Answer", use_container_width=True) and st.session_state.p1_fin:
        st.session_state.selected_button = "data_filter"
with nav_cols[2]:
    if st.button("🎓 Train", use_container_width=True) and st.session_state.p1_fin and st.session_state.p2_fin:
        st.session_state.selected_button = "train"
with nav_cols[3]:
    if st.button("📊 Eval", use_container_width=True) and st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.session_state.selected_button = "eval"


# Page Title
st.title("Data Synthesis")

# Data Generation Method Selection
method = st.selectbox("Synthesis Method", ["  ", "MAGPIE", "Self-Instruct", "Back-Translation"])

# Dynamically Display Parameters Based on Selected Method
if method != "  ":
    st.subheader(f"{method} Parameter Configuration")
    
if method == "MAGPIE":
    with st.form("magpie_config_form"):
        # Basic Configuration
        st.subheader("Basic Configuration")
        cols = st.columns(2)
        with cols[0]:
            task_name = st.text_input("Task Name", placeholder="Please enter your task name", value=st.session_state.get("magpie_task_name", ""))
        with cols[1]:
            # Timestamp Selection Box (Fixed in the second column)
            timestamp_option = st.selectbox("Timestamp Generation Method", ["Auto-generate", "Manual Input"])

        # Display the timestamp input box below the selection box
        timestamp = st.text_input("Timestamp", placeholder="Please enter the timestamp(if you chose to do it manually)", value=st.session_state.get("magpie_timestamp", ""))

        # Path Configuration
        st.subheader("Path Configuration")
        cols = st.columns(2)
        with cols[0]:
            model_path = st.text_input("Model Path", placeholder="Please enter the model path", value=st.session_state.get("magpie_model_path", ""))
        with cols[1]:
            config_path = st.text_input("Config Path", placeholder="Please enter the configuration file path", value=st.session_state.get("magpie_config_path", ""))

        # Model Configuration
        st.subheader("Model Configuration")
        model_id = st.text_input("Model ID", placeholder="Please ensure it matches the model ID on HuggingFace", value=st.session_state.get("magpie_model_id", ""))
        
        cols = st.columns(2)
        with cols[0]:
            tensor_parallel = st.number_input("Tensor Parallel", min_value=1, value=st.session_state.get("magpie_tensor_parallel", 1))
        with cols[1]:
            gpu_utilization = st.slider("GPU Utilization", 0.0, 1.0, st.session_state.get("magpie_gpu_utilization", 0.8))

        # Sampling Configuration
        st.subheader("Sampling Configuration")
        cols = st.columns(4)
        with cols[0]:
            total_prompts = st.number_input("Total Prompts", min_value=1, value=st.session_state.get("magpie_total_prompts", 100), step = 1)
        with cols[1]:
            temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=st.session_state.get("magpie_temperature", 0.7), step = 0.1)
        with cols[2]:
            top_p = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=st.session_state.get("magpie_top_p", 0.9), step = 0.1)
        with cols[3]:
            n_samples = st.number_input("N-Simultaneous Samples", min_value=1, value=st.session_state.get("magpie_n_samples", 1))

        # Device List
        st.subheader("Device List")
        devices = st.multiselect(
            "Select Devices",
            options=list(range(16)),  
            default=st.session_state.get("magpie_devices", [0]),  
            max_selections=16  
        )

        # Submit Button
        cols1, cols2, cols3 = st.columns([5, 5, 4])
        with cols2:
            saved = st.form_submit_button("💾 Save Configuration")

        if saved:
            # Check all required fields
            missing_fields = []
            if not task_name:
                missing_fields.append("Task Name")
            if timestamp_option == "Manual Input" and not timestamp:
                missing_fields.append("Timestamp")
            if not model_path:
                missing_fields.append("Model Path")
            if not config_path:
                missing_fields.append("Config Path")
            if not model_id:
                missing_fields.append("Model ID")
            if not tensor_parallel:
                missing_fields.append("Tensor Parallel")
            if not gpu_utilization:
                missing_fields.append("GPU Maximum Utilization")
            if not total_prompts:
                missing_fields.append("Total Prompts")
            if not temperature:
                missing_fields.append("Temperature")
            if not top_p:
                missing_fields.append("Top-p")
            if not devices:
                missing_fields.append("Device List")

            # If there are missing fields, prompt which fields are missing
            if missing_fields:
                st.error(f"The following fields are missing: {', '.join(missing_fields)}")
            else:
                script_content = f"""
                            model_path={model_path}
                            total_prompts={total_prompts}
                            ins_topp={top_p}
                            ins_temp={temperature}
                            config={config_path}
                            model_id={model_id}
                            res_rep=1
                            device="{','.join(map(str, devices))}"
                            tensor_parallel={tensor_parallel}
                            gpu_memory_utilization={gpu_utilization}
                            n={n_samples}

                            # Get Current Time
                            timestamp=$(date +%s)

                            # Generate Pretty Name
                            job_name="${model_path}_topp${top_p}_temp${temperature}_${timestamp}"

                            ### Setup Logging
                            log_dir="data"
                            if [ ! -d "../$log_dir" ]; then
                                mkdir -p "../$log_dir"
                            fi

                            job_path="../$log_dir/$job_name"

                            mkdir -p $job_path
                            exec > >(tee -a "$job_path/$job_name.log") 2>&1
                            echo "[magpie.sh] Model Name: $model_path"
                            echo "[magpie.sh] Pretty name: $job_name"
                            echo "[magpie.sh] Total Prompts: $total_prompts"
                            echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
                            echo "[magpie.sh] Response Generation Config: temp=$res_temp, top_p=$res_topp, rep=$res_rep"
                            echo "[magpie.sh] System Config: device=$device, n=$n, tensor_parallel=$tensor_parallel"
                            echo "[magpie.sh] Timestamp: $timestamp"
                            echo "[magpie.sh] Job Name: $job_name"

                            echo "[magpie.sh] Start Generating Instructions..."
                            CUDA_VISIBLE_DEVICES=$device python src/autoalign/data/instruction/magpie.py \\
                                --device $device \\
                                --model_path $model_path \\
                                --total_prompts $total_prompts \\
                                --top_p $ins_topp \\
                                --temperature $ins_temp \\
                                --tensor_parallel $tensor_parallel \\
                                --gpu_memory_utilization $gpu_memory_utilization \\
                                --n $n \\
                                --job_name $job_name \\
                                --timestamp $timestamp \\
                                --model-id $model_id \\
                                --config $config

                            echo "[magpie.sh] Finish Generating Instructions!"
                            """
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                bash_file_path = os.path.join(parent_dir, "magpie.sh")
                # 将脚本内容保存到文件
                with open(bash_file_path, "w") as f:
                    f.write(script_content)
                st.success(f"MAGPIE script saved successfully at: {bash_file_path}")
                st.session_state.selected_button = "data_filter"
                st.session_state.p1_fin = True
                time.sleep(2)
                st.switch_page("page2.py")
                            
# Self-Instruct Method Parameters
elif method == "Self-Instruct":
    with st.form("self_instruct_config_form"):
        # Basic Configuration
        st.subheader("Basic Configuration")
        model_id = st.text_input("Model ID", placeholder="Please enter the model name (customizable)", value=st.session_state.get("self_instruct_model_id", ""))
        template_name = st.text_input("Template", placeholder="Please enter the template name as per ATA regulations", value=st.session_state.get("self_instruct_template_name", ""))
        #根据后端添加
        output_path = st.text_input("Output Path", placeholder="Please enter the output path", value=st.session_state.get("self_instruct_output_path", ""))

        # Path Configuration
        st.subheader("Path Configuration")
        cols = st.columns(2)
        with cols[0]:
            question_gen_model_path = st.text_input("Self-Instruct Model Path", placeholder="Please enter the model path", value=st.session_state.get("self_instruct_question_gen_model_path", ""))
        with cols[1]:
            seed_data_path = st.text_input("Seed Data Path", placeholder="Please enter the data path in ShareGPT format", value=st.session_state.get("self_instruct_seed_data_path", ""))

        # Inference Configuration
        st.subheader("Inference Configuration")
        cols = st.columns(2)
        with cols[0]:
            backend = st.selectbox("Inference Backend", ["hf", "vllm"], index=0 if st.session_state.get("self_instruct_backend", "hf") == "hf" else 1)
        with cols[1]:
            num_prompts = st.number_input("Self-Instruct Count", min_value=1, value=st.session_state.get("self_instruct_num_prompts", 10))

        # Submit Button
        cols1, cols2, cols3 = st.columns([5, 5, 4])
        with cols2:
            saved = st.form_submit_button("💾 Save Configuration")

        # Validation and Processing After Submission
        if saved:
            # Check all required fields
            missing_fields = []
            if not model_id:
                missing_fields.append("Model ID")
            if not template_name:
                missing_fields.append("Template Name")
            if not question_gen_model_path:
                missing_fields.append("Self-Instruct Model Path")
            if not seed_data_path:
                missing_fields.append("Seed-data-path")
            if not backend:
                missing_fields.append("Inference Backend")
            if not num_prompts:
                missing_fields.append("Self-Instruct Count")

            # If there are missing fields, prompt which fields are missing
            if missing_fields:
                st.error(f"The following fields are missing: {', '.join(missing_fields)}")
            else:
                script_content = f"""
                                python src/autoalign/data/instruction/self_instruct.py \\
                                    --model-id {model_id} \\
                                    --template-name {template_name} \\
                                    --question-gen-model-path {question_gen_model_path} \\
                                    --seed-data-path {seed_data_path} \\
                                    --backend {backend} \\
                                    --num-prompts {num_prompts}\\
                                    --output-path {output_path}

                                """
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                bash_file_path = os.path.join(parent_dir, "self_instruct.sh")
                # 将脚本内容保存到文件
                with open(bash_file_path, "w") as f:
                    f.write(script_content)
                st.success(f"Self-Instruct script saved successfully at: {bash_file_path}")
                st.session_state.selected_button = "data_filter"
                st.session_state.p1_fin = True
                time.sleep(2)
                st.switch_page("page2.py")
        
# Back-Translation Method Parameters
elif method == "Back-Translation":
    with st.form("back_translation_config_form"):
        # Sampling Configuration
        st.subheader("Sampling Configuration")
        cols = st.columns(3)
        with cols[0]:
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=st.session_state.get("back_translation_temperature", 0.7), step = 0.1)
        with cols[1]:
            top_p = st.number_input("Top-p", min_value=0.0, max_value=1.0, value=st.session_state.get("back_translation_top_p", 0.9), step = 0.1)
        with cols[2]:
            max_length = st.number_input("Max Length", min_value=50, value=st.session_state.get("back_translation_max_length", 8192), step = 2048)

        # Path Configuration
        st.subheader("Path Configuration")
        cols = st.columns(2)
        with cols[0]:
            unlabeled_data_path = st.text_input("Unlabeled Data Path", placeholder="Please enter the unlabeled data path", value=st.session_state.get("back_translation_unlabeled_data_path", ""))
        with cols[1]:
            output_path = st.text_input("Output File Path", placeholder="Please enter the output file save path", value=st.session_state.get("back_translation_output_path", ""))
        prompt_column_name = st.text_input("prompt_column_name", placeholder="Please enter the prompt column name", value=st.session_state.get("back_translation_prompt_column_name", ""))
        model_path = st.text_input("Back-Translation Model Path", placeholder="Please enter the back-translation model path", value=st.session_state.get("back_translation_model_path", ""))

        st.subheader("Inference Configuration")
        cols = st.columns(2)
        with cols[0]:
            tensor_parallel_size = st.number_input("Tensor Parallel Size", min_value=1,  value=st.session_state.get("back_translation_tensor_parallel_size", 1), help="Number of GPUs each model uses for inference")
        with cols[1]:
            devices = st.multiselect(
            "Select Devices",
            options=list(range(16)),  
            default=st.session_state.get("magpie_devices", [0]),  
            max_selections=16  
        )
            devices_str = ','.join(map(str, devices))

    
        # Submit Button
        cols1, cols2, cols3 = st.columns([5, 5, 4])
        with cols2:
            saved = st.form_submit_button("💾 Save Configuration")

        # Validation and Processing After Submission
        if saved:
            # Check all required fields
            missing_fields = []
            if not temperature:
                missing_fields.append("Temperature")
            if not top_p:
                missing_fields.append("Top-p")
            if not max_length:
                missing_fields.append("Max Length")
            if not prompt_column_name:
                missing_fields.append("prompt_column_name")
            if not devices:
                missing_fields.append("devices")
            if not unlabeled_data_path:
                missing_fields.append("Unlabeled Data Path")
            if not output_path:
                missing_fields.append("Output File Save Path")
            if not model_path:
                missing_fields.append("Back-Translation Model Path")
            if not tensor_parallel_size:
                missing_fields.append("Tensor Parallel Size")

            # If there are missing fields, prompt which fields are missing
            if missing_fields:
                st.error(f"The following fields are missing: {', '.join(missing_fields)}")
            else:
                script_content = f"""
                                #!/usr/bin/bash


                                export CUDA_VISIBLE_DEVICES={devices_str}

                                model_path={model_path}
                                data_filepath={unlabeled_data_path}
                                save_filepath={output_path}
                                prompt_column_name={prompt_column_name}

                                python src/autoalign/data/instruction/back_translation.py \\
                                    --reverse \\
                                    --model_path={model_path} \\
                                    --data_filepath={unlabeled_data_path} \\
                                    --save_filepath={output_path} \\
                                    --prompt_column_name={prompt_column_name} \\
                                    --tensor_parallel_size={tensor_parallel_size}
                                """
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(current_dir)
                bash_file_path = os.path.join(parent_dir, "back_translation.sh")
                # 将脚本内容保存到文件
                with open(bash_file_path, "w") as f:
                    f.write(script_content)
                st.success(f"Back-Translation script saved successfully at: {bash_file_path}")
                st.session_state.selected_button = "data_filter"
                st.session_state.p1_fin = True
                time.sleep(2)
                st.switch_page("page2.py")



if st.session_state.selected_button == "data_gen":
    pass
elif st.session_state.selected_button == "data_filter":
    if st.session_state.p1_fin:
        st.switch_page("page2.py")
    else:
        st.title("Please finish this page first")
        st.session_state.selected_button = "data_gen"
elif st.session_state.selected_button == "train":
    if st.session_state.p1_fin and st.session_state.p2_fin:
        st.switch_page("page3.py")
elif st.session_state.selected_button == "eval":
    if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.switch_page("page4.py")


    
