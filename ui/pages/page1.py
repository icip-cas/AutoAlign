import streamlit as st
import os
import time
from pages.navbar import render_navbar, check_and_switch_page_1, init_session_state
import pages.navbar

render_navbar()

# 页面标题
st.title("Data Synthesis")

# 使用 session_state 保存 method 的选择状态
if "selected_method" not in st.session_state:
    st.session_state["selected_method"] = "  "

# 选择框
method = st.selectbox(
    "Synthesis Method",
    ["  ", "MAGPIE", "Self-Instruct", "Back-Translation"],
    index=["  ", "MAGPIE", "Self-Instruct", "Back-Translation"].index(
        st.session_state["selected_method"]
    ),
)

# 更新 session_state 中的 method
st.session_state["selected_method"] = method

# 动态显示参数配置
if method != "  ":
    st.subheader(f"{method} Parameter Configuration")

# MAGPIE 配置
if method == "MAGPIE":
    with st.form("magpie_config_form"):
        # Basic Configuration
        st.subheader("Basic Configuration")
        cols = st.columns(3)
        with cols[0]:
            task_name = st.text_input(
                "Task Name",
                placeholder="Please enter your task name",
                value=st.session_state.get("magpie_task_name", ""),
            )
        with cols[1]:
            # Timestamp Selection Box (Fixed in the second column)
            timestamp_option = st.selectbox(
                "Timestamp Generation Method",
                ["Auto-generate", "Manual Input"],
                index=0
                if st.session_state.get("magpie_timestamp_option", "Auto-generate")
                == "Auto-generate"
                else 1,
            )
        with cols[2]:
            # Display the timestamp input box below the selection box
            timestamp = st.text_input(
                "Timestamp",
                placeholder="Please enter the timestamp(if you chose to do it manually)",
                value=st.session_state.get("magpie_timestamp", ""),
            )

        # Path Configuration
        st.subheader("Path Configuration")
        cols = st.columns(2)
        with cols[0]:
            model_path = st.text_input(
                "Model Path",
                placeholder="Please enter the model path",
                value=st.session_state.get("magpie_model_path", ""),
            )
        with cols[1]:
            config_path = st.text_input(
                "Config Path",
                placeholder="Please enter the configuration file path",
                value=st.session_state.get("magpie_config_path", ""),
            )

        # Model Configuration
        st.subheader("Model Configuration")
        cols = st.columns(3)
        with cols[0]:
            model_id = st.text_input(
                "Model ID",
                placeholder="Please ensure it matches the model ID on HuggingFace",
                value=st.session_state.get("magpie_model_id", ""),
            )
        with cols[1]:
            tensor_parallel = st.number_input(
                "Tensor Parallel",
                min_value=1,
                value=st.session_state.get("magpie_tensor_parallel", 1),
            )
        with cols[2]:
            gpu_utilization = st.slider(
                "GPU Utilization",
                0.0,
                1.0,
                st.session_state.get("magpie_gpu_utilization", 0.8),
            )

        # Sampling Configuration
        st.subheader("Sampling Configuration")
        cols = st.columns(4)
        with cols[0]:
            total_prompts = st.number_input(
                "Total Prompts",
                min_value=1,
                value=st.session_state.get("magpie_total_prompts", 100),
                step=1,
            )
        with cols[1]:
            temperature = st.number_input(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("magpie_temperature", 0.7),
                step=0.1,
            )
        with cols[2]:
            top_p = st.number_input(
                "Top-p",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("magpie_top_p", 0.9),
                step=0.1,
            )
        with cols[3]:
            n_samples = st.number_input(
                "N-Simultaneous Samples",
                min_value=1,
                value=st.session_state.get("magpie_n_samples", 1),
            )

        # Device List
        st.subheader("Device List")
        devices = st.multiselect(
            "Select Devices",
            options=list(range(16)),
            default=st.session_state.get("magpie_devices", [0]),
            max_selections=16,
        )

        # Submit Button
        cols1, cols2, cols3 = st.columns([1, 1, 1])
        with cols2:
            saved = st.form_submit_button("→")

        if saved:
            # Save all inputs to session_state
            st.session_state["Syn_method"] = "MAGPIE"
            st.session_state["magpie_task_name"] = task_name
            st.session_state["magpie_timestamp_option"] = timestamp_option
            st.session_state["magpie_timestamp"] = timestamp
            st.session_state["magpie_model_path"] = model_path
            st.session_state["magpie_config_path"] = config_path
            st.session_state["magpie_model_id"] = model_id
            st.session_state["magpie_tensor_parallel"] = tensor_parallel
            st.session_state["magpie_gpu_utilization"] = gpu_utilization
            st.session_state["magpie_total_prompts"] = total_prompts
            st.session_state["magpie_temperature"] = temperature
            st.session_state["magpie_top_p"] = top_p
            st.session_state["magpie_n_samples"] = n_samples
            st.session_state["magpie_devices"] = devices

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

            # 检查路径合法性
            invalid_paths = []
            if model_path and not os.path.exists(model_path):
                invalid_paths.append(f"Model Path '{model_path}' does not exist.")
            if config_path and not os.path.exists(config_path):
                invalid_paths.append(f"Config Path '{config_path}' does not exist.")

            # 如果有缺失字段或无效路径，显示详细的错误信息
            if missing_fields or invalid_paths:
                error_message = "Please fix the following errors before saving:\n\n"

                if missing_fields:
                    error_message += "**Missing Fields:**\n"
                    for field in missing_fields:
                        error_message += f"- {field} is required.\n"

                if invalid_paths:
                    error_message += "\n**Invalid Paths:**\n"
                    for path_error in invalid_paths:
                        error_message += f"- {path_error}\n"

                st.error(error_message)
            else:
                script_content = f"""
model_path={model_path}
total_prompts={total_prompts}
ins_topp={top_p}
ins_temp={temperature}
config={config_path}
model_id={model_id}
device="{",".join(map(str, devices))}"
tensor_parallel={tensor_parallel}
gpu_memory_utilization={gpu_utilization}
n={n_samples}

# Get Current Time
timestamp=$(date +%s)

# Generate Pretty Name
job_name="${model_id}_topp${top_p}_temp${temperature}_${timestamp}"#job_name="${task_name}_topp${top_p}_temp${temperature}_${timestamp}"

### Setup Logging
log_dir="data"
if [ ! -d "./$log_dir" ]; then
    mkdir -p "./$log_dir"
fi

job_path="./$log_dir/$job_name"

mkdir -p $job_path
exec > >(tee -a "$job_path/$job_name.log") 2>&1
echo "[magpie.sh] Model Name: $model_path"
echo "[magpie.sh] Pretty name: $job_name"
echo "[magpie.sh] Total Prompts: $total_prompts"
echo "[magpie.sh] Instruction Generation Config: temp=$ins_temp, top_p=$ins_topp"
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
                st.session_state.step1 = script_content
                st.success("Configuration Saved!")
                st.session_state.p1_fin = True
                time.sleep(1.5)
                st.session_state.selected_button = "data_filter"
                st.switch_page("pages/page2.py")


elif method == "Self-Instruct":
    with st.form("self_instruct_config_form"):
        # Basic Configuration
        st.subheader("Basic Configuration")
        cols = st.columns(3)
        with cols[0]:
            model_id = st.text_input(
                "Model ID",
                placeholder="Please enter the model name (customizable)",
                value=st.session_state.get("self_instruct_model_id", ""),
            )
        with cols[1]:
            template_name = st.text_input(
                "Template",
                placeholder="Please enter the template name as per ATA regulations",
                value=st.session_state.get("self_instruct_template_name", ""),
            )
        with cols[2]:
            output_path = st.text_input(
                "Output Path",
                placeholder="Please enter the output path",
                value=st.session_state.get("self_instruct_output_path", ""),
            )

        # Path Configuration
        st.subheader("Path Configuration")
        cols = st.columns(2)
        with cols[0]:
            question_gen_model_path = st.text_input(
                "Self-Instruct Model Path",
                placeholder="Please enter the model path",
                value=st.session_state.get("self_instruct_question_gen_model_path", ""),
            )
        with cols[1]:
            seed_data_path = st.text_input(
                "Seed Data Path",
                placeholder="Please enter the data path in ShareGPT format",
                value=st.session_state.get("self_instruct_seed_data_path", ""),
            )

        # Inference Configuration
        st.subheader("Inference Configuration")
        cols = st.columns(2)
        with cols[0]:
            backend = st.selectbox(
                "Inference Backend",
                ["hf", "vllm"],
                index=0
                if st.session_state.get("self_instruct_backend", "hf") == "hf"
                else 1,
            )
        with cols[1]:
            num_prompts = st.number_input(
                "Self-Instruct Count",
                min_value=1,
                value=st.session_state.get("self_instruct_num_prompts", 10),
            )

        # Submit Button
        cols1, cols2, cols3 = st.columns([1, 1, 1])
        with cols2:
            # saved = st.form_submit_button("💾 Save Configuration")
            saved = st.form_submit_button("→")

        # Validation and Processing After Submission
        if saved:
            # Save all inputs to session_state
            st.session_state["Syn_method"] = "Self-Instruct"
            st.session_state["self_instruct_model_id"] = model_id
            st.session_state["self_instruct_template_name"] = template_name
            st.session_state["self_instruct_output_path"] = output_path
            st.session_state["self_instruct_question_gen_model_path"] = (
                question_gen_model_path
            )
            st.session_state["self_instruct_seed_data_path"] = seed_data_path
            st.session_state["self_instruct_backend"] = backend
            st.session_state["self_instruct_num_prompts"] = num_prompts

            # Check all required fields
            missing_fields = []
            if not model_id:
                missing_fields.append("Model ID")
            if not template_name:
                missing_fields.append("Template Name")
            if not question_gen_model_path:
                missing_fields.append("Self-Instruct Model Path")
            if not seed_data_path:
                missing_fields.append("Seed Data Path")
            if not backend:
                missing_fields.append("Inference Backend")
            if not num_prompts:
                missing_fields.append("Self-Instruct Count")

            # 检查路径合法性
            invalid_paths = []
            if question_gen_model_path and not os.path.exists(question_gen_model_path):
                invalid_paths.append(
                    f"Self-Instruct Model Path '{question_gen_model_path}' does not exist."
                )
            if seed_data_path and not os.path.exists(seed_data_path):
                invalid_paths.append(
                    f"Seed Data Path '{seed_data_path}' does not exist."
                )

            # 如果有缺失字段或无效路径，显示详细的错误信息
            if missing_fields or invalid_paths:
                error_message = "Please fix the following errors before saving:\n\n"

                if missing_fields:
                    error_message += "**Missing Fields:**\n"
                    for field in missing_fields:
                        error_message += f"- {field} is required.\n"

                if invalid_paths:
                    error_message += "\n**Invalid Paths:**\n"
                    for path_error in invalid_paths:
                        error_message += f"- {path_error}\n"

                st.error(error_message)
            else:
                script_content = f"""
python src/autoalign/data/instruction/self_instruct.py \\
    --job-id {model_id} \\
    --template-name {template_name} \\
    --question-gen-model-path {question_gen_model_path} \\
    --seed-data-path {seed_data_path} \\
    --backend {backend} \\
    --num-prompts {num_prompts}\\
    --output-path {output_path}
"""
                # current_dir = os.path.dirname(os.path.abspath(__file__))
                # parent_dir = os.path.dirname(current_dir)
                # parent_dir = parent_dir +"/sc/"
                # if not os.path.exists(parent_dir):
                #     os.makedirs(parent_dir)
                # bash_file_path = os.path.join(parent_dir, "self_instruct.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                # st.success(f"Self-Instruct script saved successfully at: {bash_file_path}")
                st.session_state.step1 = script_content
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.session_state.p1_fin = True
                st.session_state.selected_button = "data_filter"
                st.switch_page("pages/page2.py")

elif method == "Back-Translation":
    with st.form("back_translation_config_form"):
        st.subheader("Path Configuration")
        cols = st.columns(2)
        with cols[0]:
            unlabeled_data_path = st.text_input(
                "Unlabeled Data Path",
                placeholder="Please enter the unlabeled data path",
                value=st.session_state.get("back_translation_unlabeled_data_path", ""),
            )
        with cols[1]:
            output_path = st.text_input(
                "Output File Path",
                placeholder="Please enter the output file save path",
                value=st.session_state.get("back_translation_output_path", ""),
            )
        prompt_column_name = st.text_input(
            "prompt_column_name",
            placeholder="Please enter the prompt column name",
            value=st.session_state.get("back_translation_prompt_column_name", ""),
        )
        model_path = st.text_input(
            "Back-Translation Model Path",
            placeholder="Please enter the back-translation model path",
            value=st.session_state.get("back_translation_model_path", ""),
        )

        st.subheader("Inference Configuration")
        cols = st.columns(2)
        with cols[0]:
            tensor_parallel_size = st.number_input(
                "Tensor Parallel Size",
                min_value=1,
                value=st.session_state.get("back_translation_tensor_parallel_size", 1),
                help="Number of GPUs each model uses for inference",
            )
        with cols[1]:
            devices = st.multiselect(
                "Select Devices",
                options=list(range(16)),
                default=st.session_state.get("magpie_devices", [0]),
                max_selections=16,
            )
            devices_str = ",".join(map(str, devices))

        # Submit Button
        cols1, cols2, cols3 = st.columns([1, 1, 1])
        with cols2:
            # saved = st.form_submit_button("💾 Save Configuration")
            saved = st.form_submit_button("→")

        # Validation and Processing After Submission
        if saved:
            # Save all inputs to session_state
            # st.session_state["back_translation_temperature"] = temperature
            # st.session_state["back_translation_top_p"] = top_p
            # st.session_state["back_translation_max_length"] = max_length
            st.session_state["Syn_method"] = "Back-Translation"
            st.session_state["back_translation_unlabeled_data_path"] = (
                unlabeled_data_path
            )
            st.session_state["back_translation_output_path"] = output_path
            st.session_state["back_translation_prompt_column_name"] = prompt_column_name
            st.session_state["back_translation_model_path"] = model_path
            st.session_state["back_translation_tensor_parallel_size"] = (
                tensor_parallel_size
            )
            st.session_state["back_translation_devices"] = devices

            # Check all required fields
            missing_fields = []
            # if not temperature:
            #     missing_fields.append("Temperature")
            # if not top_p:
            #     missing_fields.append("Top-p")
            # if not max_length:
            #     missing_fields.append("Max Length")
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

            # Check path validity
            invalid_paths = []
            if unlabeled_data_path and not os.path.exists(unlabeled_data_path):
                invalid_paths.append(
                    f"Unlabeled Data Path '{unlabeled_data_path}' does not exist."
                )
            if model_path and not os.path.exists(model_path):
                invalid_paths.append(
                    f"Back-Translation Model Path '{model_path}' does not exist."
                )

            # If there are missing fields or invalid paths, display detailed error messages
            if missing_fields or invalid_paths:
                error_message = "Please fix the following errors before saving:\n\n"

                if missing_fields:
                    error_message += "**Missing Fields:**\n"
                    for field in missing_fields:
                        error_message += f"- {field} is required.\n"

                if invalid_paths:
                    error_message += "\n**Invalid Paths:**\n"
                    for path_error in invalid_paths:
                        error_message += f"- {path_error}\n"

                st.error(error_message)
            else:
                script_content = f"""
#!/usr/bin/bash

export CUDA_VISIBLE_DEVICES={devices_str}

model_path={model_path}
data_filepath={unlabeled_data_path}
save_filepath={output_path}
prompt_column_name={prompt_column_name}
tensor_parallel_size={tensor_parallel_size}

python src/autoalign/data/instruction/back_translation.py \\
    --reverse \\
    --model_path={model_path} \\
    --data_filepath={unlabeled_data_path} \\
    --save_filepath={output_path} \\
    --prompt_column_name={prompt_column_name} \\
    --tensor_parallel_size={tensor_parallel_size}
"""
                # current_dir = os.path.dirname(os.path.abspath(__file__))
                # parent_dir = os.path.dirname(current_dir)
                # parent_dir = parent_dir +"/sc/"
                # if not os.path.exists(parent_dir):
                #     os.makedirs(parent_dir)
                # bash_file_path = os.path.join(parent_dir, "back_translation.sh")
                # # 将脚本内容保存到文件
                # with open(bash_file_path, "w") as f:
                #     f.write(script_content)
                # st.success(f"Back-Translation script saved successfully at: {bash_file_path}")
                st.session_state.step1 = script_content
                st.success("Configuration Saved!")
                time.sleep(1.5)
                st.session_state.p1st.session_state.p1_fin = True
                st.session_state.selected_button = "data_filter"
                st.switch_page("pages/page2.py")

if st.session_state.selected_button == "data_gen":
    print("/////////////////////////////////////")
    pass
elif st.session_state.selected_button == "data_filter":
    if st.session_state.p1_fin:
        st.switch_page("pages/page2.py")
    else:
        st.error("Please finish this page first")
        st.session_state.selected_button = "data_gen"
elif st.session_state.selected_button == "train":
    if st.session_state.p1_fin and st.session_state.p2_fin:
        st.switch_page("pages/page3.py")
elif st.session_state.selected_button == "eval":
    if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.switch_page("pages/page4.py")
