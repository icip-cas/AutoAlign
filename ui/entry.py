import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/page1.py"),
        st.Page("pages/page2.py"),
        st.Page("pages/page3.py"),
        st.Page("pages/page4.py"),
        st.Page("pages/loading.py"),
        st.Page("pages/page5.py"),
        st.Page("pages/page6.py"),
        st.Page("pages/page7.py"),
        st.Page("pages/page8.py"),
    ]
)
pg.run()
