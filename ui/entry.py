import streamlit as st
pg = st.navigation([st.Page("page1.py"), st.Page("page2.py")])
pg.run()