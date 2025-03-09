import streamlit as st
pg = st.navigation([st.Page("1.query_generate.py"), st.Page("4.eval.py"), st.Page("8.eval_visua.py")])
pg.run()