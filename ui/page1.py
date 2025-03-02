import os
import streamlit as st
from transformers import AutoTokenizer
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
from conversation import TEMPLATES, Role, Conversation
import uuid
from io import StringIO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


n_gram = 4
# 绘图函数
def draw_bin_graph(data, num_bins, title, x, y):
    counts, bin_edges = np.histogram(data, bins=num_bins)
    fig = go.Figure(data=[go.Bar(
        x=bin_edges[:-1],
        y=counts,
        width=np.diff(bin_edges),
        marker=dict(color='skyblue')
    )])
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, showlegend=False)
    st.plotly_chart(fig)

def draw_pie_graph(data, title):
    labels, num = zip(*data)
    total = sum(num)
    percentages = [f"{label} - {n} ({n / total * 100:.1f}%)" for label, n in zip(labels, num)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=num, hoverinfo='label+value+percent', textinfo='label+percent', text=percentages)])
    fig.update_layout(title=title, showlegend=True)
    st.plotly_chart(fig)

def draw_sunburst_graph(df, title):
    df = df.nlargest(10, 'count')
    fig = px.sunburst(df, path=['verb', 'noun'], values='count')
    fig.update_layout(title=title)
    st.plotly_chart(fig)


# 加载数据

def pro_file(file_path, data_type, toeknizer):
    with open(file_path) as f:
        ids = []
        file_name, file_extension = os.path.splitext(file_path)
        if file_extension == '.json':
            data = json.load(f)
        elif file_extension == '.jsonl':
            data = []
            for line in f:
                s_line = line.strip()
                if s_line:
                    try:
                        data.append(json.loads(s_line))
                    except json.JSONDecodeError:
                        st.error(f"invalid json {line}")
                        continue
        if data_type == "sft":
            turn_num = []
            sources = []
            token_num = []
            texts = []
            domain = []
            data_time = []

            for item in data:
                conversations = item['conversations']
                turn = len(conversations) // 2
                if "dom" in item:
                    dom = item['dom']
                else:
                    dom = 'unknown'
                if "time" in item:
                    time = item["time"]
                else:
                    time = 'unknown'
                if st.session_state.get('chat_template', 'default') == 'default':
                    token = len(tokenizer.apply_chat_template(conversations))
                else:
                    conv = Conversation(template=TEMPLATES[st.session_state.chat_template])
                    conv.fill_in_messages(item)
                    tk_conv = conv.get_tokenized_conversation(tokenizer, 5000)
                    token = len(tk_conv['input_ids'])
                
                
                ids.append(item['id'])
                texts.append(conversations)
                sources.append(item.get('source', "unknown"))
                turn_num.append(turn)
                token_num.append(token)
                domain.append(dom)
                data_time.append(time)

            df = pd.DataFrame({
                "ID": ids,
                "Text": texts,
                "Source": sources,
                "Turns": turn_num,
                "Tokens": token_num,
                "Domain": domain,
                "Time": data_time
            })
        elif data_type == "dpo":
            smooth = SmoothingFunction().method7
            self_bleu_scores = []
            domain = []
            data_time = []
            for i in range(len(data)):
                candi = str(data[i]).split()
                ref = [str(data).split() for j, text in enumerate(data) if j != i]
                score = sentence_bleu(ref, candi, weights=(1/n_gram,)*n_gram, smoothing_function=smooth)
                self_bleu_scores.append(score)
            group_score = sum(self_bleu_scores)/len(self_bleu_scores)
            for item in data:
                if "dom" in item:
                    dom = item['dom']
                else:
                    dom = 'unknown'
                if "time" in item:
                    time = item["time"]
                else:
                    time = 'unknown'
                domain.append(dom)
                data_time.append(time)

            df = pd.DataFrame({
                "text": data,
                "Score": group_score,
                "Domain": domain,
                "Time": data_time
            })
        return df, data
        
def pro_folder(folder_path, data_type, tokenizer):
    all_dfs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            df, data = pro_file(file_path, data_type, tokenizer)
            if df is not None:
                all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        st.error("No valid files found in the folder")
        return None

# 隐藏侧边栏
hide_sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
"""
# st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# 初始化 session_state 变量
for key, default in {
    "mod_dir": '',
    "file_dir": '',
    "submit_clicked": False,
    "show_turn": False,
    "show_source": False,
    "show_token": False,
    "show_time": False,
    "chat_template": '',
    "data_type": "",
    "show_domain": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

data_type = st.selectbox(
    'Type_select',
    (
        'sft', 'dpo'
    )
)

st.session_state.data_type = data_type
if data_type == "sft":
    with st.form('input_dir'):
        st.session_state.mod_dir = st.text_input('Model Directory:')
        st.session_state.file_dir = st.text_input('File Directory')
        st.session_state.show_turn = st.checkbox('Show Turn Distribution')
        st.session_state.show_source = st.checkbox('Show Source Distribution')
        st.session_state.show_token = st.checkbox('Show Token Distribution')
        st.session_state.show_time = st.checkbox('Show Time Distribution')
        st.session_state.show_domain = st.checkbox('Show Domain Distribution')
        st.session_state.chat_template = st.selectbox(
            'Template Select',
            ("chatml", "vicuna_v1.1", "llama-2-chat", "llama-2-chat-keep-system", 
            "chatml-keep-system", "llama-3-instruct", "mistral-instruct", 
            "gemma", "zephyr", "chatml-idsys", "glm-4-chat", 
            "glm-4-chat-keep-system", "default")
        )
        submit = st.form_submit_button('Submit')
        if submit:
            st.session_state.submit_clicked = True

elif data_type == "dpo":
    with st.form('input_dir'):
        st.session_state.mod_dir = st.text_input('Model Directory:')
        st.session_state.file_dir = st.text_input('File Directory')
        st.session_state.show_domain = st.checkbox('Show Domain Distribution')
        st.session_state.show_time = st.checkbox('Show Time Distribution')
        submit = st.form_submit_button('Submit')
        if submit:
            st.session_state.submit_clicked = True

# 获取用户输入的目录
mod_dir = str(st.session_state.mod_dir)
file_dir = str(st.session_state.file_dir)

# 验证模型和文件路径
if st.session_state.submit_clicked:
    mod_available = True
    path_available = os.path.exists(file_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(mod_dir)
    except Exception:
        mod_available = False
        if submit:
            st.write('Invalid model directory!')

    if not path_available:
        if submit:
            st.write('Invalid data directory!')

    if mod_available and path_available:
        wide_window_style = """
        <style>
        #root > div:nth-child(1) > div.withScreencast > div > div > div > section.stMain.st-emotion-cache-bm2z3a.ea3mdgi8 > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 {
            max-width: 90% !important;
        }
        </style>
        """
        st.markdown(wide_window_style, unsafe_allow_html=True)
        tokenizer = AutoTokenizer.from_pretrained(st.session_state.mod_dir)
        file_path = st.session_state.file_dir
        data_type = st.session_state.data_type
        if data_type == "sft":
            if os.path.isfile(file_path):
                df, data = pro_file(file_path, data_type, tokenizer)
            elif os.path.isdir(file_path):
                df, data = pro_folder(file_path, data_type, tokenizer )
            st.sidebar.header("Filter")
            selected_sources = st.sidebar.multiselect(
                "Data Source",
                options=df['Source'].unique(),
                default=df['Source'].unique()
            )

            selected_domain = st.sidebar.multiselect(
                "Data Domain",
                options=df['Domain'].unique(),
                default=df['Domain'].unique()
            )


            min_turns, max_turns = st.sidebar.slider(
                "Turn Distribution",
                min_value=int(df['Turns'].min()),
                max_value=int(df['Turns'].max())+1,
                value=(int(df['Turns'].min()), int(df['Turns'].max()))
            )

            min_tokens, max_tokens = st.sidebar.slider(
                "Token Distribution",
                min_value=int(df['Tokens'].min()),
                max_value=int(df['Tokens'].max())+1,
                value=(int(df['Tokens'].min()), int(df['Tokens'].max()))
            )

            filtered_df = df[
                (df['Source'].isin(selected_sources)) &
                (df['Turns'].between(min_turns, max_turns)) &
                (df['Tokens'].between(min_tokens, max_tokens))
            ]
        elif data_type == "dpo":
            if os.path.isfile(file_path):
                df, data = pro_file(file_path, data_type, tokenizer)
            elif os.path.isdir(file_path):
                df, data = pro_folder(file_path, data_type, tokenizer )
            st.sidebar.header("Filter")
            min_score, max_score = st.sidebar.slider(
                "Score Distribution",
                min_value=(df['Score'].min()),
                max_value=(df['Score'].max())+0.1,
                value=((df['Score'].min()), (df['Score'].max()))
            )
            filtered_df = df[
                (df['Score'].between(min_score, max_score)) 
            ]


        st.title("Data Explorer")


        
        PAGE_SIZE = 6
        if 'page' not in st.session_state:
            st.session_state.page = 0

        # total_pages = len(filtered_df) // PAGE_SIZE + (1 if len(filtered_df) % PAGE_SIZE else 0)

        # def update_page(change):
        #     st.session_state.page = max(0, min(st.session_state.page + change, total_pages-1))

        if data_type == "sft":
            total_pages = len(filtered_df) // PAGE_SIZE + (1 if len(filtered_df) % PAGE_SIZE else 0)

            def update_page(change):
                st.session_state.page = max(0, min(st.session_state.page + change, total_pages-1))

            with st.container():

                cols = st.columns([1,1,1])
                with cols[0]: 
                    st.markdown(f'''
                    <div class="metric-box">
                        <div style="font-size:14px;color:#666;">Total</div>
                        <div style="font-size:18px;color:#2e86c1;">{len(filtered_df):,}</div>
                    </div>''', unsafe_allow_html=True)
                
                with cols[1]:
                    st.markdown(f'''
                    <div class="metric-box">
                        <div style="font-size:14px;color:#666;">Filtered</div>
                        <div style="font-size:18px;color:#28b463;">{len(filtered_df):,}</div>
                    </div>''', unsafe_allow_html=True)
                
                with cols[2]:
                    avg_tokens = int(filtered_df['Tokens'].mean())
                    st.markdown(f'''
                    <div class="metric-box">
                        <div style="font-size:14px;color:#666;">Avg Tokens</div>
                        <div style="font-size:18px;color:#d35400;">{avg_tokens:,}</div>
                    </div>''', unsafe_allow_html=True)
                
                st.write("")

                # 分页表格
                start_idx = st.session_state.page * PAGE_SIZE
                end_idx = (st.session_state.page + 1) * PAGE_SIZE
                page_data = filtered_df.iloc[start_idx:end_idx]

                st.dataframe(
                    page_data,
                    height=300,
                    use_container_width=True,
                    column_config={
                        "Text": st.column_config.TextColumn(
                            "Conversation",
                            width="large",
                            help="First 50 characters",
                        )
                    }
                )

                # 分页控件
                col1, col2, col3 = st.columns([2,5,2])
                with col2:
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:10px;">
                        <span style="color:#666;">当前 {start_idx + 1}-{min(end_idx, len(filtered_df))} 条 / 共 {len(filtered_df)} 条</span>
                        <button class="pagination-btn" onclick="arguments[0].stopPropagation(); updatePage(-1)" {'disabled' if st.session_state.page == 0 else ''}>上一页</button>
                        <span style="margin:0 10px;color:#333;">{st.session_state.page + 1}/{total_pages}</span>
                        <button class="pagination-btn" onclick="arguments[0].stopPropagation(); updatePage(1)" {'disabled' if st.session_state.page == total_pages -1 else ''}>下一页</button>
                    </div>
                    """, unsafe_allow_html=True)
            

            st.components.v1.html(f"""
            <script>
            function updatePage(change) {{
                parent.postMessage({{type: 'session', method: 'callMethod', args: ['update_page', change]}}, '*');
            }}
            </script>
            """)

            # 绘制数据统计结果
            source_count = filtered_df['Source'].tolist()
            res_source_count = list(Counter(source_count).items()) if source_count else []
            domain_count = filtered_df['Domain'].tolist()
            res_dom_count = list(Counter(domain_count).items()) if domain_count else []
            time_count = filtered_df['Time'].tolist()
            res_time_count = list(Counter(time_count).items()) if time_count else []
            # 动态绘制各图表
            graphs = [
                (draw_bin_graph, {'data': filtered_df['Turns'].tolist(), 'num_bins': 10, 'title': 'Turn Distribution', 'x': 'Turn', 'y': 'Count'}, st.session_state.show_turn),
                (draw_pie_graph, {'data': res_source_count, 'title': 'Source Distribution'}, st.session_state.show_source),
                (draw_bin_graph, {'data': filtered_df['Tokens'].tolist(), 'num_bins': 10, 'title': 'Token Distribution', 'x': 'Token', 'y': 'Count'}, st.session_state.show_token),
                (draw_pie_graph, {'data': res_dom_count, 'title': 'Source Domain'}, st.session_state.show_domain),
                (draw_pie_graph, {'data': res_time_count, 'title': 'Time Distribution'}, st.session_state.show_time)

            ]

            columns = st.columns(3)
            curr_col = 0
            for func, params, draw in graphs:
                if draw:
                    with columns[curr_col]:
                        try:
                            func(**params)
                        except Exception as e:
                            pass
                    curr_col = (curr_col + 1) % 3
        elif data_type == "dpo":
            total_pages = len(filtered_df) // PAGE_SIZE + (1 if len(filtered_df) % PAGE_SIZE else 0)

            def update_page(change):
                st.session_state.page = max(0, min(st.session_state.page + change, total_pages-1))

            with st.container():

                cols = st.columns([1])
                with cols[0]:   
                    st.markdown(f'''
                    <div class="metric-box">
                        <div style="font-size:14px;color:#666;">Total</div>
                        <div style="font-size:18px;color:#2e86c1;">{len(filtered_df):,}</div>
                    </div>''', unsafe_allow_html=True)
                
                
                st.write("")

                # 分页表格
                start_idx = st.session_state.page * PAGE_SIZE
                end_idx = (st.session_state.page + 1) * PAGE_SIZE
                page_data = filtered_df.iloc[start_idx:end_idx]

                st.dataframe(
                    page_data,
                    height=300,
                    use_container_width=True,
                    column_config={
                        "Text": st.column_config.TextColumn(
                            "Conversation",
                            width="large",
                            help="First 50 characters",
                        )
                    }
                )

                # 分页控件
                col1, col2, col3 = st.columns([2,5,2])
                with col2:
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:10px;">
                        <span style="color:#666;">当前 {start_idx + 1}-{min(end_idx, len(filtered_df))} 条 / 共 {len(filtered_df)} 条</span>
                        <button class="pagination-btn" onclick="arguments[0].stopPropagation(); updatePage(-1)" {'disabled' if st.session_state.page == 0 else ''}>上一页</button>
                        <span style="margin:0 10px;color:#333;">{st.session_state.page + 1}/{total_pages}</span>
                        <button class="pagination-btn" onclick="arguments[0].stopPropagation(); updatePage(1)" {'disabled' if st.session_state.page == total_pages -1 else ''}>下一页</button>
                    </div>
                    """, unsafe_allow_html=True)
            

            st.components.v1.html(f"""
            <script>
            function updatePage(change) {{
                parent.postMessage({{type: 'session', method: 'callMethod', args: ['update_page', change]}}, '*');
            }}
            </script>
            """)
            domain_count = filtered_df['Domain'].tolist()
            res_dom_count = list(Counter(domain_count).items()) if domain_count else []
            time_count = filtered_df['Time'].tolist()
            res_time_count = list(Counter(time_count).items()) if time_count else []
            graphs = [
                (draw_pie_graph, {'data': res_dom_count, 'title': 'Source Domain'}, st.session_state.show_domain),
                (draw_pie_graph, {'data': res_time_count, 'title': 'Time Distribution'}, st.session_state.show_time)
            ]

            columns = st.columns(3)
            curr_col = 0
            for func, params, draw in graphs:
                if draw:
                    with columns[curr_col]:
                        try:
                            func(**params)
                        except Exception as e:
                            pass
                    curr_col = (curr_col + 1) % 3



        # # 数据导出功能
        # if st.button("export"):
        #     # 转换为原始数据格式
        #     export_data = []
        #     for _, row in filtered_df.iterrows():
        #         original_item = next(item for item in data if item['id'] == row['ID'])
        #         export_data.append(original_item)
            
        #     buffer = StringIO()
        #     for item in export_data:
        #         buffer.write(json.dumps(item) + "\n")
            
        #     st.download_button(
        #         label="download",
        #         data=buffer.getvalue(),
        #         file_name=f"filtered_data_{uuid.uuid4().hex[:6]}.jsonl",
        #         mime="application/jsonl+json"
        #     )