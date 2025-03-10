import streamlit as st
st.set_page_config(layout="centered")
nav_cols = st.columns(4)
for key, default in {
    "p1_fin": False,
    "p2_fin": False,
    "p3_fin": False,
    "p4_fin": False,
    "selected_button": "data_gen"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

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

import streamlit as st

st.title("Model Eval")

with st.form("config_form"):
    st.subheader("BenchMark")
    col1, col2 = st.columns([3,2])
    with col1:
        process = st.selectbox(
            "BenchMark", 
            [
            "MATH",           		 
            "GSM-8K",         		   
            "HumanEval",      		
            "MBPP",       		    
            "HumanEval-CN",
            "MBPP-CN",	   
            "MMLU",        		
            "GPQA",        		    
            "CMMLU",       		    
            "C-Eval",
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

if st.session_state.selected_button == "data_gen":
    st.switch_page("page1.py")
elif st.session_state.selected_button == "data_filter":
    st.switch_page("page2.py")
elif st.session_state.selected_button == "train":
    st.switch_page("page3.py")
elif st.session_state.selected_button == "eval":
    pass