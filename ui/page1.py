import streamlit as st

# Page Title
st.title("Data Synthesis")

# Data Generation Method Selection
method = st.selectbox("Synthesis Method", ["  ", "MAGPIE", "Self-Instruct", "Back-Translation"])

# Dynamically Display Parameters Based on Selected Method
if method != "  ":
    st.subheader(f"{method} Parameter Configuration")
    
# MAGPIE Method Parameters
if method == "MAGPIE":
    with st.form("magpie_config_form"):
        # Basic Configuration
        st.subheader("Basic Configuration")
        cols = st.columns(2)
        with cols[0]:
            task_name = st.text_input("Task Name", placeholder="Please enter your task name", value=st.session_state.get("magpie_task_name", ""))
        with cols[1]:  # Needs modification
            # Timestamp Selection Box (Fixed in the second column)
            timestamp_option = st.selectbox("Timestamp Generation Method", ["Auto-generate", "Manual Input"])

        timestamp_cols = st.columns(2)  # Create a two-column layout
        # If manual input is selected, generate an input box spanning two columns
        if timestamp_option == "Manual Input":
            with timestamp_cols[0]:  # Occupies the first column
                timestamp = st.text_input("Manual Input Timestamp", placeholder="Please enter the timestamp", value=st.session_state.get("magpie_timestamp", ""))
            with timestamp_cols[1]:  # Occupies the second column (left empty for layout purposes)
                pass
        else:
            # Default to auto-generating timestamp
            timestamp = "Auto-generated timestamp"
            st.text_input("Auto-generated Timestamp", value=timestamp, disabled=True)

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
            options=["GPU 0", "GPU 1", "GPU 2", "GPU 3","GPU 4", "GPU 5", "GPU 6", "GPU 7"],
            default=st.session_state.get("magpie_devices", ["GPU 0"])
        )

        # Submit Button
        cols1, cols2, cols3 = st.columns([5, 5, 4])
        with cols2:
            submitted = st.form_submit_button("🚀 Submit Configuration")

        # Validation and Processing After Submission
        if submitted:
            # Check all required fields
            missing_fields = []
            if not task_name:
                missing_fields.append("Task Name")
            if not timestamp:
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
                # Save configuration to session_state
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
                st.success("Configuration saved successfully!")
                # Redirect to page3.py
                st.switch_page("page3.py")
    
# Self-Instruct Method Parameters
elif method == "Self-Instruct":
    with st.form("self_instruct_config_form"):
        # Basic Configuration
        st.subheader("Basic Configuration")
        model_id = st.text_input("Model ID", placeholder="Please enter the model name (customizable)", value=st.session_state.get("self_instruct_model_id", ""))
        template_name = st.text_input("Template", placeholder="Please enter the template name as per ATA regulations", value=st.session_state.get("self_instruct_template_name", ""))

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
            backend = st.selectbox("Inference Backend", ["HF", "Vllm"], index=0 if st.session_state.get("self_instruct_backend", "hf") == "hf" else 1)
        with cols[1]:
            num_prompts = st.number_input("Self-Instruct Count", min_value=1, value=st.session_state.get("self_instruct_num_prompts", 10))

        # Submit Button
        cols1, cols2, cols3 = st.columns([5, 5, 4])
        with cols2:
            submitted = st.form_submit_button("🚀 Submit Configuration")

        # Validation and Processing After Submission
        if submitted:
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
                # Save configuration to session_state
                st.session_state.self_instruct_config = {
                    "model_id": model_id,
                    "template_name": template_name,
                    "question_gen_model_path": question_gen_model_path,
                    "seed_data_path": seed_data_path,
                    "backend": backend,
                    "num_prompts": num_prompts
                }
                st.success("Configuration saved successfully!")
                # Redirect to page3.py
                st.switch_page("page3.py")
        
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
        back_translation_model_path = st.text_input("Back-Translation Model Path", placeholder="Please enter the back-translation model path", value=st.session_state.get("back_translation_model_path", ""))

        st.subheader("Inference Configuration")
        tensor_parallel_size = st.number_input("Tensor Parallel Size", min_value=1,  value=st.session_state.get("back_translation_tensor_parallel_size", 1), help="Number of GPUs each model uses for inference")

        # Submit Button
        cols1, cols2, cols3 = st.columns([5, 5, 4])
        with cols2:
            submitted = st.form_submit_button("🚀 Submit Configuration")

        # Validation and Processing After Submission
        if submitted:
            # Check all required fields
            missing_fields = []
            if not temperature:
                missing_fields.append("Temperature")
            if not top_p:
                missing_fields.append("Top-p")
            if not max_length:
                missing_fields.append("Max Length")
            if not unlabeled_data_path:
                missing_fields.append("Unlabeled Data Path")
            if not output_path:
                missing_fields.append("Output File Save Path")
            if not back_translation_model_path:
                missing_fields.append("Back-Translation Model Path")
            if not tensor_parallel_size:
                missing_fields.append("Tensor Parallel Size")

            # If there are missing fields, prompt which fields are missing
            if missing_fields:
                st.error(f"The following fields are missing: {', '.join(missing_fields)}")
            else:
                # Save configuration to session_state
                st.session_state.back_translation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_length": max_length,
                    "unlabeled_data_path": unlabeled_data_path,
                    "output_path": output_path,
                    "back_translation_model_path": back_translation_model_path,
                    "tensor_parallel_size": tensor_parallel_size
                }
                st.success("Configuration saved successfully!")
                # Redirect to page3.py
                st.switch_page("page3.py")