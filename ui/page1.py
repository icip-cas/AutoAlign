import streamlit as st
st.markdown("<span style='color: blue;'>数据生成</span> ———————— 数据筛选 ———————— 训练 ———————— 评测", unsafe_allow_html=True)
intel = st.text_input("请输入文字")
submit = st.button("Submit")
if submit:
    if intel == "":
        st.write("请输入文字！")
    else:
        st.switch_page("page2.py")