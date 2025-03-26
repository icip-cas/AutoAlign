import streamlit as st
<<<<<<< Updated upstream

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
=======
pg = st.navigation([st.Page("query_generate_1.py"), st.Page("eval_4.py")])
pg.run()
>>>>>>> Stashed changes
