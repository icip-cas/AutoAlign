import streamlit as st
pg = st.navigation([st.Page("page1.py"), st.Page("page2.py"), st.Page("page3.py"), st.Page("page4.py")])
pg.run()