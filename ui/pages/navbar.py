import streamlit as st


def render_navbar():
    # 初始化会话状态
    for key, default in {
        "p1_fin": False,
        "p2_fin": False,
        "p3_fin": False,
        "p4_fin": False,
        "selected_button": "data_gen",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # 自定义 CSS 样式
    st.markdown(
        """
        <style>
            div.stButton > button,
            div.stFormSubmitButton > button,
            .nav-button {
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
            div.stFormSubmitButton > button:focus,
            .nav-button:hover,
            .nav-button:active,
            .nav-button:focus {
                background: #1976D2 !important;
                color: white !important;  /* 强制保持白色 */
                transform: scale(1.05);
                box-shadow: none !important;  /* 移除聚焦阴影 */
                outline: none !important;     /* 移除聚焦轮廓 */
            }

            /* 强制禁用所有颜色变换 */
            div.stButton > button:hover span,
            div.stButton > button:active span,
            div.stButton > button:focus span,
            .nav-button:hover span,
            .nav-button:active span,
            .nav-button:focus span {
                color: inherit !important;  /* 继承父级颜色 */
            }
            
            .btn-text {
                font-size: 1.5rem;
                text-align: center;
                margin-top: 5px;
            }
            
            .current-page-button {
                transform: scale(1.2) !important;
                font-size: 2.4rem !important;
            }
            
            .completed-button {
                background: #2196F3 !important;
            }
            
            .incomplete-button {
                background: #808080 !important;
            }

            /* 横杠样式 */
            .separator {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            .separator hr {
                width: 100%;
                margin: 0;
                border: 2px solid #808080;
                border-radius: 2px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 页面布局
    nav_cols = st.columns(7)
    # 定义导航栏的按钮和标签
    labels = [
        ("Generate Query", "I", "data_gen", "page1"),
        ("", "", "", ""),
        ("Sample Answer", "II", "data_filter", "page2"),
        ("", "", "", ""),
        ("Train", "III", "train", "page3"),
        ("", "", "", ""),
        ("Eval", "IV", "eval", "page4"),
    ]

    # 渲染导航栏
    nav_cols = st.columns(7)
    for i, (text, num, key, page) in enumerate(labels):
        with nav_cols[i]:
            if key and page:
                is_current_page = st.session_state["selected_button"] == key

                # 判断是否已完成
                is_completed = False
                if key == "data_gen":
                    is_completed = st.session_state["p1_fin"]
                elif key == "data_filter":
                    is_completed = st.session_state["p2_fin"]
                elif key == "train":
                    is_completed = st.session_state["p3_fin"]
                elif key == "eval":
                    is_completed = st.session_state["p4_fin"]

                # 动态设置按钮样式
                button_class = "nav-button"
                if is_current_page:
                    button_class += " current-page-button"
                if is_completed:
                    button_class += " completed-button"
                else:
                    button_class += " incomplete-button"

                st.markdown(
                    f"""
                    <div class="{button_class}" 
                            onclick="window.parent.postMessage('{key}', '*');">
                        {num}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 渲染按钮下方的文字
                st.markdown(
                    f"<div style='text-align: center; margin-top: 10px;'>{text}</div>",
                    unsafe_allow_html=True,
                )
            else:
                # 判断横线是否需要变色
                if i == 1:
                    is_completed = st.session_state["p1_fin"]
                elif i == 3:
                    is_completed = st.session_state["p2_fin"]
                elif i == 5:
                    is_completed = st.session_state["p3_fin"]

                # 动态设置横线颜色
                line_color = "#2196F3" if is_completed else "#808080"
                st.markdown(
                    f"""
                    <div class='hr-line' 
                            style='background: {line_color}; 
                                height: 4px; 
                                width: 100%; 
                                margin-left: 2%; 
                                margin-right: 20%; 
                                margin-top: calc(min(6vw, 80px) / 2 - 2px); 
                                border-radius: 2px;'>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# 这部分确认跳转到哪个页面,并且要判断是否允许调整，4页面有4哥页面的对应的判断


def check_and_switch_page_1():
    if st.session_state.selected_button == "data_gen":
        pass
    elif st.session_state.selected_button == "data_filter":
        if st.session_state.p1_fin:
            st.switch_page("page2.py")
        else:
            st.error("Please finish this page first")
            st.session_state.selected_button = "data_gen"
    elif st.session_state.selected_button == "train":
        if st.session_state.p1_fin and st.session_state.p2_fin:
            st.switch_page("page3.py")
    elif st.session_state.selected_button == "eval":
        if (
            st.session_state.p1_fin
            and st.session_state.p2_fin
            and st.session_state.p3_fin
        ):
            st.switch_page("page4.py")


def check_and_switch_page_2():
    if st.session_state.selected_button == "data_gen":
        st.switch_page("page1.py")
    elif st.session_state.selected_button == "data_filter":
        pass
    elif st.session_state.selected_button == "train":
        if st.session_state.p1_fin and st.session_state.p2_fin:
            st.switch_page("page3.py")
        else:
            st.title("Please finish this page first")
            st.session_state.selected_button == "data_filter"
    elif st.session_state.selected_button == "eval":
        if (
            st.session_state.p1_fin
            and st.session_state.p2_fin
            and st.session_state.p3_fin
        ):
            st.switch_page("page4.py")


def check_and_switch_page_3():
    if st.session_state.selected_button == "data_gen":
        st.switch_page("page1.py")
    elif st.session_state.selected_button == "data_filter":
        st.switch_page("page2.py")
    elif st.session_state.selected_button == "train":
        pass
    elif st.session_state.selected_button == "eval":
        if (
            st.session_state.p1_fin
            and st.session_state.p2_fin
            and st.session_state.p3_fin
        ):
            st.switch_page("page4.py")


def check_and_switch_page_4():
    if st.session_state.selected_button == "data_gen":
        st.switch_page("page1.py")
    elif st.session_state.selected_button == "data_filter":
        st.switch_page("page2.py")
    elif st.session_state.selected_button == "train":
        st.switch_page("page3.py")
    elif st.session_state.selected_button == "eval":
        pass


def init_session_state():
    if "p1_fin" not in st.session_state:
        st.session_state.p1_fin = False
    if "p2_fin" not in st.session_state:
        st.session_state.p2_fin = False
    if "p3_fin" not in st.session_state:
        st.session_state.p3_fin = False
    if "p4_fin" not in st.session_state:
        st.session_state.p4_fin = False

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = "data_gen"


def render_navbar_visual():
    for key, default in {"selected_button": "data_demo"}.items():
        if key not in st.session_state:
            st.session_state[key] = default
    st.markdown(
        """
        <style>
            div.stButton > button,
            div.stFormSubmitButton > button,
            .nav-button {
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
            div.stFormSubmitButton > button:focus,
            .nav-button:hover,
            .nav-button:active,
            .nav-button:focus {
                background: #1976D2 !important;
                color: white !important;  /* 强制保持白色 */
                transform: scale(1.05);
                box-shadow: none !important;  /* 移除聚焦阴影 */
                outline: none !important;     /* 移除聚焦轮廓 */
            }

            /* 强制禁用所有颜色变换 */
            div.stButton > button:hover span,
            div.stButton > button:active span,
            div.stButton > button:focus span,
            .nav-button:hover span,
            .nav-button:active span,
            .nav-button:focus span {
                color: inherit !important;  /* 继承父级颜色 */
            }
            
            .btn-text {
                font-size: 1.5rem;
                text-align: center;
                margin-top: 5px;
            }
            
            .current-page-button {
                transform: scale(1.2) !important;
                font-size: 2.4rem !important;
            }
            
            .completed-button {
                background: #2196F3 !important;
            }
            
            .incomplete-button {
                background: #808080 !important;
            }

            /* 横杠样式 */
            .separator {
                display: flex;
                align-items: center;
                justify-content: center;
                height: 100%;
            }
            .separator hr {
                width: 100%;
                margin: 0;
                border: 2px solid #808080;
                border-radius: 2px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    labels = [
        ("Data Board", "V", "data_gen", "page5"),
        ("", "", "", ""),
        ("Logs", "VI", "data_filter", "page6"),
        ("", "", "", ""),
        ("Train", "VII", "train", "page7"),
        ("", "", "", ""),
        ("Eval", "VIII", "eval", "page8"),
    ]

    # 渲染导航栏
    nav_cols = st.columns(7)
    for i, (text, num, key, page) in enumerate(labels):
        with nav_cols[i]:
            if key:
                # 动态设置按钮样式
                button_class = "nav-button"
                button_class += " completed-button"
                with st.container():
                    if st.button(num, key=key):
                        st.session_state["selected_button"] = key
                    # st.markdown(
                    #     f"<div class='btn-text'>{text}</div>", unsafe_allow_html=True
                    # )

                    # 渲染按钮下方的文字
                    st.markdown(
                        f"<div style='text-align: center; margin-top: 10px;'>{text}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                line_color = "#2196F3"
                st.markdown(
                    f"""
                    <div class='hr-line' 
                            style='background: {line_color}; 
                                height: 4px; 
                                width: 100%; 
                                margin-left: 2%; 
                                margin-right: 20%; 
                                margin-top: calc(min(6vw, 80px) / 2 - 2px); 
                                border-radius: 2px;'>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
