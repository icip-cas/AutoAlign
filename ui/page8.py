import streamlit as st
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("Data Board", use_container_width=True):
        st.session_state.selected_button = "data_demo"
with nav_cols[1]:
    if st.button("Logs", use_container_width=True):
        st.session_state.selected_button = "logs"
with nav_cols[2]:
    if st.button("Training", use_container_width=True) and st.session_state.p2_fin:
        st.session_state.selected_button = "training"
with nav_cols[3]:
    if st.button("Benchmark", use_container_width=True) and st.session_state.p2_fin and st.session_state.p3_fin:
        st.session_state.selected_button = "benchmark"

if st.session_state.selected_button == "data_demo":
    st.switch_page("page5.py")
elif st.session_state.selected_button == "logs":
    st.switch_page("page6.py")
elif st.session_state.selected_button == "training":
    st.switch_page("page7.py")
elif st.session_state.selected_button == "benchmark":
    pass