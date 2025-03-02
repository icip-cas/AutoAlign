import streamlit as st
pg = st.navigation([st.Page("page1.py", title="可视化"), st.Page("page2.py", title="数据生成"), st.Page("page3.py", title="训练")])
pg.run()