import streamlit as st
import time
for key, default in {
    "p1_fin": False,
    "p2_fin": False,
    "p3_fin": False,
    "p4_fin": False,
    "selected_button": "data_gen"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
st.set_page_config(layout="centered")
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("📥 Generate Query", use_container_width=True):
        st.session_state.selected_button = "data_gen"
with nav_cols[1]:
    if st.button("🔎 Sample Answer", use_container_width=True):
        st.session_state.selected_button = "data_filter"
with nav_cols[2]:
    if st.button("🎓 Train", use_container_width=True) and st.session_state.p2_fin:
        st.session_state.selected_button = "train"
with nav_cols[3]:
    if st.button("📊 Eval", use_container_width=True) and st.session_state.p2_fin and st.session_state.p3_fin:
        st.session_state.selected_button = "eval"

if "sub_sel_button" not in st.session_state:
    st.session_state.sub_sel_button = "SFT"
st.session_state.sub_sel_button = st.selectbox(
    "Configure: ",
    ("SFT", "SPIN", "SSO", "RLCD", "CAI", "Self_Rewarding")
)
with st.form("Train Configuration"):
    if st.session_state.sub_sel_button == 'SFT':
            cols = st.columns([1, 1])
            with cols[0]: 
                proj_name = st.text_input("Project Name")
                model = st.text_input("Base Model")
                dataset = st.text_input("Dataset Source")
                output_dir = st.text_input("Output Dir")
            with cols[1]:
                columns = st.columns([1, 1, 1])
                with columns[0]:
                    train_batch_size = st.number_input("Train Batch Size per Device", value=4)
                    chat_template = st.selectbox(
                        'Template Select',
                        ("chatml", "vicuna_v1.1", "llama-2-chat", "llama-2-chat-keep-system", 
                        "chatml-keep-system", "llama-3-instruct", "mistral-instruct", 
                        "gemma", "zephyr", "chatml-idsys", "glm-4-chat", 
                        "glm-4-chat-keep-system", "default")
                    )
                    eval_strategy = st.selectbox(
                        "Eval Strategy",
                        ("no")
                    )
                    bf16 = st.selectbox(
                        'bf16',
                        (
                            'True', 'False'
                        )
                    )
                    lr_scheduler = st.selectbox(
                        "LR Scheduler",
                        ("cosine")
                    )
                    logging_steps = st.number_input("Logging Steps", value=1)
                    deepspeed = st.text_input("Deepspeed", value="Deepspeed_dir")
                    eval_num = st.number_input("Eval Num", value=0)

                with columns[1]:
                    learning_rate = st.text_input("Learning Rate", value="2e-5")
                    num_train_epoch = st.number_input("Train Epoch", value=3)
                    eval_batch_size = st.number_input("Eval Batch Size", value=4)
                    eval_steps = st.number_input("Eval Steps", value=1500)
                    weight_decay = st.number_input("Weight Decay", value=0)
                    report_to = st.selectbox(
                        "Report To",
                        ("Tensorboard")
                    )
                    ddp_timeout = st.number_input('DDP Timeout', value=18000)
                    num_workers = st.number_input('Worker Num', value=1)
                with columns[2]:
                    model_max_length = st.number_input('Model Max Length', value=4096)
                    
                    gradient_accumulation_steps = st.number_input("GA", value=1)
                    save_strategy = st.selectbox(
                        'Save Strategy',
                        (
                            'epoch'
                        )
                    )
                    warmup_ratio = st.number_input("Warmup Ratio", value=-0.04)
                    logging_dir = st.text_input("Logging Dir")
                    gradient_checkpoint = st.selectbox(
                        "Gradient Checkpoint",
                        (True, False)
                    )
                    lazy_preprocess = st.selectbox(
                        "Lazy Preprocess",
                        (True, False)
                    )
    elif st.session_state.sub_sel_button == 'SPIN':
        cols = st.columns([1, 1])
        with cols[0]: 
            proj_name = st.text_input("Project Name")
            model = st.text_input("Base Model")
            dataset = st.text_input("Dataset Source")
            output_dir = st.text_input("Output Dir")
        with cols[1]:
            columns = st.columns([1, 1])
            with columns[0]:
                iter_rounds = st.number_input("Iteration Rounds")
                threshold = st.number_input("Thresholds")
                data_source = st.text_input("Instruction Source")
            with columns[1]:
                temperature = st.number_input("Temperature")
                top_p = st.number_input("Top-p Sampling")
                prompt_template = st.text_input("Prompt")


    elif st.session_state.sub_sel_button == 'SSO':
        cols = st.columns([1, 1])
        with cols[0]: 
            proj_name = st.text_input("Project Name")
            model = st.text_input("Base Model")
            dataset = st.text_input("Dataset Source")
            output_dir = st.text_input("Output Dir")
        with cols[1]:
            columns = st.columns([1, 1, 1])
            with columns[0]:
                train_batch_size = st.number_input("Train Batch Size per Device", value=4)
                chat_template = st.selectbox(
                    'Template Select',
                    ("chatml", "vicuna_v1.1", "llama-2-chat", "llama-2-chat-keep-system", 
                    "chatml-keep-system", "llama-3-instruct", "mistral-instruct", 
                    "gemma", "zephyr", "chatml-idsys", "glm-4-chat", 
                    "glm-4-chat-keep-system", "default")
                )
                eval_strategy = st.selectbox(
                    "Eval Strategy",
                    ("no")
                )
                bf16 = st.selectbox(
                    'bf16',
                    (
                        'True', 'False'
                    )
                )
                lr_scheduler = st.selectbox(
                    "LR Scheduler",
                    ("cosine")
                )
                logging_steps = st.number_input("Logging Steps", value=1)
                deepspeed = st.text_input("Deepspeed", value="Deepspeed_dir")
                
            with columns[1]:
                learning_rate = st.text_input("Learning Rate", value="2e-5")
                beta = st.number_input("Beta")
                num_train_epoch = st.number_input("Train Epoch", value=3)
                eval_batch_size = st.number_input("Eval Batch Size", value=4)
                eval_steps = st.number_input("Eval Steps", value=1500)
                weight_decay = st.number_input("Weight Decay", value=0)
                report_to = st.selectbox(
                        "Report To",
                        ("Tensorboard")
                    )
                with columns[2]:    
                    gradient_accumulation_steps = st.number_input("GA", value=1)
                    save_strategy = st.selectbox(
                        'Save Strategy',
                        (
                            'epoch'
                        )
                    )
                    warmup_ratio = st.number_input("Warmup Ratio", value=-0.04)
                    logging_dir = st.text_input("Logging Dir")
                    gradient_checkpoint = st.selectbox(
                        "Gradient Checkpoint",
                        (True, False)
                    )
            

    elif st.session_state.sub_sel_button == 'RLCD':
        cols = st.columns([1, 1])
        with cols[0]: 
            proj_name = st.text_input("Project Name")
            model = st.text_input("Base Model")
            dataset = st.text_input("Dataset Source")
            output_dir = st.text_input("Output Dir")
        with cols[1]:
            columns = st.columns([1, 1])
            with columns[0]:
                data_source = st.text_input("Instruction Source")
                response_control_prompt = st.text_area("Response Control Prompt")
            with columns[1]:
                temperature = st.number_input("Temperature")
                top_p = st.number_input("Top-p Sampling")
                prompt_template = st.text_input("Prompt")

    elif st.session_state.sub_sel_button == 'CAI':
        cols = st.columns([1, 1])
        with cols[0]: 
            proj_name = st.text_input("Project Name")
            model = st.text_input("Base Model")
            dataset = st.text_input("Dataset Source")
            output_dir = st.text_input("Output Dir")
        with cols[1]:
            columns = st.columns([1, 1])
            with columns[0]:
                red_attack_source = st.text_input("Red Team Attack Data Source")
                helpful_instruction_source = st.text_input("Helpful Instruction Data Source")
                judge_num = st.number_input("Number of Generated Judges")
                avh_ratio = st.text_input("Attack vs Helpful Data Ratio")
            with columns[1]:
                temperature = st.number_input("Temperature")
                top_p = st.number_input("Top-p Sampling")
                prompt_template = st.text_input("Prompt")
                critique_iterations = st.number_input("Critique Iterations")

    elif st.session_state.sub_sel_button == 'Self_Rewarding':
        cols = st.columns([1, 1])
        with cols[0]: 
            proj_name = st.text_input("Project Name")
            model = st.text_input("Base Model")
            dataset = st.text_input("Dataset Source")
            output_dir = st.text_input("Output Dir")
        with cols[1]:
            columns = st.columns([1, 1])
            with columns[0]:
                iteration_rounds = st.number_input("Iteration Rounds")
                eft = st.selectbox(
                    "Evaluation Fine Tuning",
                    ("yes", "no")
                )
                self_ins_path = st.text_input("Self Instruct Model Path")
                ift_source = st.text_input("Instruction Fine-Tuning Data Source")
                eft_source = st.text_input("Evaluation Fine-Tuning Data Source")
                sim_thresh = st.number_input("Similarity Threshold for Instruction Deduplication")
            with columns[1]:
                temperature = st.number_input("Temperature")
                top_p = st.number_input("Top-p Sampling")
                prompt_template = st.text_input("Prompt")
                instruction_per_iter = st.number_input("Instruction Num per Iteration")
                bon_per_iter = st.number_input("BoN responces per Iteration")
                judge_iter = st.number_input("Judges per Iteration")

    if st.form_submit_button("Submit"):
        # Temporary skipping part
        st.session_state.p3_fin = True
        st.success("Successfully Saved!")
        time.sleep(0.7)
        st.session_state.selected_button = "eval"
        st.switch_page("page4.py")


if st.session_state.selected_button == "data_gen":
    st.switch_page("page1.py")
elif st.session_state.selected_button == "data_filter":
    st.switch_page("page2.py")
elif st.session_state.selected_button == "train":
    pass
elif st.session_state.selected_button == "eval":
    if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.switch_page("page4.py")





