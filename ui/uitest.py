import streamlit as st

st.markdown("""
    <style>
        div.stButton > button {
            width: 55px;
            height: 55px;
            border-radius: 50%;
            background: #2196F3;
            color: white;
            border: none;
            cursor: pointer;
            transition: 0.3s;
            font-size: 1.6rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
        }
        div.stButton > button:hover {
            background: #1976D2;
            transform: scale(1.05);
        }S
        .btn-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
        }
        .btn-text {
            font-size: 1rem;
            text-align: center;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# 创建 7 列布局
nav_cols = st.columns(7)
labels = [
    ("Generate Query", "1", "data_gen"),
    ("", "", ""),
    ("Sample Answer", "2", "data_filter"),
    ("", "", ""),
    ("Train", "3", "train"),
    ("", "", ""),
    ("Eval", "4", "eval")
]


for i, (text, num, key) in enumerate(labels):
    with nav_cols[i]:
        if key:
            with st.container():
                if st.button(num, key=key):
                    st.session_state['selected_button'] = key
                st.markdown(f"<div class='btn-text'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='hr-line' style='background: #2196F3; height: 4px; width: 100%; margin: 28px auto 0; border-radius: 2px;'></div>", unsafe_allow_html=True)