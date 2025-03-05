import streamlit as st
st.markdown("<span style='color: blue;'>数据生成 ———————— 数据筛选 ———————— 训练 ———————— 评测</span>", unsafe_allow_html=True)

st.title("Model Eval")

with st.form("config_form"):
    st.subheader("BenchMark")
    col1, col2 = st.columns([3,2])
    with col1:
        process = st.selectbox(
            "BenchMark", 
            [
            "1",
            "2",
            "3"
            ],
            label_visibility="collapsed"
        )
    
    st.subheader("Model Selection")

    model_dir = st.text_input("Model Dir", placeholder = "Please provide the path for the model.", label_visibility="collapsed")

    st.subheader("Output Path")

    output_dir = st.text_input("Result Dir", placeholder="Please specify the path for saving the results.", label_visibility="collapsed")

    col1, col2, col3  = st.columns([4, 2, 4])
    with col2:
        submitted = st.form_submit_button("🚀 Start")

    if submitted:
        all_fields_filled = True

        if not model_dir:
            st.error("Please provide the model directory.")
            all_fields_filled = False

        if not output_dir:
            st.error("Please specify the output directory.")
            all_fields_filled = False

        if all_fields_filled:
            st.switch_page("page5.py")
        