import streamlit as st

st.markdown("<span style='color: blue;'>数据生成</span> ———————— 数据筛选 ———————— 训练 ———————— 评测", unsafe_allow_html=True)

st.title("Data Synthesis")

with st.form("config_form"):
    col1, col2 = st.columns([2, 3])
    with col1:
        process = st.selectbox("Synthesis Method", ["Magpie","Self-instruction", "Instruction Backtranslation"])
    
    
    st.subheader("Path Selection")
    data_source = st.text_input("Input Dir", placeholder="Please provide the path for the data.")

    model_dir = st.text_input("Model Dir", placeholder = "Please provide the path for the model.")

    output_dir = st.text_input("Output Dir", placeholder="Please specify the path for saving the results.")

    st.subheader("Parameters")
    cols = st.columns(3)
    with cols[0]:
        project_name = st.number_input("Temperature", min_value = 0.0, max_value = 1.0, value = 0.8, step = 0.1)
    with cols[1]:
        iteration = st.number_input("Iteration Rounds", min_value=1, max_value=100, value=10)
    with cols[2]:
        temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

  
    submitted = st.form_submit_button("Submit")

    if submitted:
        all_fields_filled = True
        if not data_source:
            st.error("Please provide the input directory.")
            all_fields_filled = False

        if not model_dir:
            st.error("Please provide the model directory.")
            all_fields_filled = False

        if not output_dir:
            st.error("Please specify the output directory.")
            all_fields_filled = False

        if not project_name:
            st.error("Please provide the project name.")
            all_fields_filled = False

        if all_fields_filled:
            st.switch_page("page2.py")
            