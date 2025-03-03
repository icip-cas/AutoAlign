import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/instruction_generation_ui.py"),
        st.Page("pages/sampling_answer_ui.py"),
        st.Page("pages/training_ui.py"),
        st.Page("pages/evaluation_ui.py"),
        st.Page("pages/loading_pages_ui.py"),
        st.Page("pages/gen_visualization_ui.py"),
        st.Page("pages/res_visualization_ui.py"),
        st.Page("pages/train_visualization_ui.py"),
        st.Page("pages/eval_visualization_ui.py"),
        st.Page("pages/iteration_setting.py")
    ]
)
pg.run()