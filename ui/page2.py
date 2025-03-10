import streamlit as st
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

with st.form("Sample Answer"):
    temperature = st.number_input("Temperature")
    top_p = st.number_input("Top-p Sampling")
    top_k = st.number_input("Top-k Sampling")
    max_new_token = st.number_input("Max New Token")

    if st.form_submit_button("Submit"):
        if not all([temperature, top_p, top_k, max_new_token]):
            st.error("Please fill in all required messages")
        else:
            st.session_state.p2_fin = True
            st.success("Successfully saved!")


if st.session_state.selected_button == "data_gen":
    st.switch_page("page1.py")
elif st.session_state.selected_button == "data_filter":
    pass
elif st.session_state.selected_button == "train":
    if st.session_state.p1_fin and st.session_state.p2_fin:
        st.switch_page("page3.py")
    else:
        st.title("Please finish this page first")
        st.session_state.selected_button == "data_filter"
elif st.session_state.selected_button == "eval":
    if st.session_state.p1_fin and st.session_state.p2_fin and st.session_state.p3_fin:
        st.switch_page("page4.py")