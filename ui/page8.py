import streamlit as st
import pandas as pd
import time
for key, default in {
    "selected_button": "data_demo"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default
st.markdown("""
    <style>
        div.stButton > button,
        div.stFormSubmitButton > button {
            width: min(6vw, 80px);
            height: min(6vw, 80px);
            border-radius: 50%;
            background: #2196F3;
            color: white !important;  /* 强制文字颜色 */
            border: none;
            cursor: pointer;
            transition: 0.3s;
            font-size: 2rem !important;
            font-weight: bold !important;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
        }
            
        div.stButton > button > div > p,
        div.stFormSubmitButton > button > div > p {
            font-size: 2rem;    
        }

        /* 覆盖所有交互状态 */
        div.stButton > button:hover,
        div.stButton > button:active,
        div.stButton > button:focus,
        div.stFormSubmitButton > button:hover,
        div.stFormSubmitButton > button:active,
        div.stFormSubmitButton > button:focus {
            background: #1976D2 !important;
            color: white !important;  /* 强制保持白色 */
            transform: scale(1.05);
            box-shadow: none !important;  /* 移除聚焦阴影 */
            outline: none !important;     /* 移除聚焦轮廓 */
        }

        /* 强制禁用所有颜色变换 */
        div.stButton > button:hover span,
        div.stButton > button:active span,
        div.stButton > button:focus span {
            color: inherit !important;  /* 继承父级颜色 */
        }
        
        .btn-text {
            font-size: 1.5rem;
            text-align: center;
            margin-top: 5px;
        }
        
    </style>
""", unsafe_allow_html=True)


nav_cols = st.columns(7)
labels = [
    ("Data Board", "5", "data_demo"),
    ("", "", ""),
    ("Logs", "6", "logs"),
    ("", "", ""),
    ("Training", "7", "training"),
    ("", "", ""),
    ("Benchmark", "8", "benchmark")
]


for i, (text, num, key) in enumerate(labels):
    with nav_cols[i]:
        if key:
            with st.container():
                if st.button(num, key=key):
                    st.session_state['selected_button'] = key
                st.markdown(f"<div class='btn-text'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='hr-line' style='background: #2196F3; height: 4px; width: 100%; margin-top: calc(min(6vw, 80px) / 2 - 2px); border-radius: 2px;'></div>", unsafe_allow_html=True
                )

st.title("Benchmark Results")

st.markdown("---")  

st.header("The evaluation progress of the benchmark")


progress_value = 0
progress_bar = st.progress(progress_value) 
status_text = st.empty()  

# 模拟进度更新
for i in range(101):
    time.sleep(0.02) 
    progress_bar.progress(i)  
    status_text.text(f"Current Progress: {i}%")  

# 任务完成提示
st.success("The benchmark evaluation is complete！")

# 添加一些空白空间，使进度条区域更大
st.markdown("<br>", unsafe_allow_html=True)  # 使用 HTML 换行增加空间

# 下半部分：多列两行的空表格
st.header("Result")

columns = [
    "GSM-8K", "HumanEval", "MBPP", "HumanEval-CN",
    "MBPP-CN", "MMLU", "GPQA", "CMMLU", "C-Eval"
]

data = {col: [""] for col in columns}  
df = pd.DataFrame(data)

df.index = ["score"]  

st.dataframe(df, height=50)  # 设置表格高度

# 添加一些样式美化界面
st.markdown("""
<style>
    /* 调整进度条的高度和颜色 */
    .stProgress > div > div > div {
        height: 25px;
        background-color: #4CAF50; /* 绿色进度条 */
    }
    /* 调整表格的边框和字体 */
    .stDataFrame {
        border: 1px solid #ccc;
        border-radius: 5px;
        font-size: 16px;
    }
    /* 调整任务完成提示的样式 */
    .stSuccess {
        font-size: 18px;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

if st.session_state.selected_button == "data_demo":
    st.switch_page("page5.py")
elif st.session_state.selected_button == "logs":
    st.switch_page("page6.py")
elif st.session_state.selected_button == "training":
    st.switch_page("page7.py")
elif st.session_state.selected_button == "benchmark":
    pass