import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from collections import Counter
from transformers import AutoTokenizer
from conversation import TEMPLATES, Role, Conversation
import uuid
from io import StringIO

# 样式设计
hide_pages_nav_style = """
<style>
    /* 通过CSS选择器定位导航容器并隐藏 */
    div[data-testid="stSidebarNav"] {
        display: none;
    }
</style>
"""
wide_window_style = """
    <style>
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section.stMain.st-emotion-cache-bm2z3a.ea3mdgi8 > div.stMainBlockContainer.block-container.st-emotion-cache-13ln4jf.ea3mdgi5 {
        max-width: 90% !important;
    }
    </style>
"""

st.markdown(hide_pages_nav_style, unsafe_allow_html=True)
st.markdown(wide_window_style, unsafe_allow_html=True)

# 页面导航
col1, col2 = st.columns(2)
with col1:
    if st.button("Return"):
        st.switch_page("page1.py")

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

# 设置模型路径和文件路径默认值
tokenizer = AutoTokenizer.from_pretrained(st.session_state.mod_dir)
file_path = st.session_state.file_dir
data_type = st.session_state.data_type
# 加载数据
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
        data_num = len(data)
        turn_num = []
        sources = []
        token_num = []
        texts = []

        for item in data:
            conversations = item['conversations']
            turn = len(conversations) // 2

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

        df = pd.DataFrame({
            "ID": ids,
            "Text": texts,
            "Source": sources,
            "Turns": turn_num,
            "Tokens": token_num
        })
    elif data_type == "dpo":
        prompt_ids = []
        prompts = []
        scores_chosen = []
        scores_rejected = []
        for item in data:
            prompt_id = item["prompt_id"]
            prompt = item["prompt"]
            score_chosen = item["score_chosen"]
            score_rejected = item["score_rejected"]

            prompt_ids.append(prompt_id)
            prompts.append(prompt)
            scores_chosen.append(score_chosen)
            scores_rejected.append(score_rejected)

        df = pd.DataFrame({
            "ID": prompt_ids,
            "Prompt": prompts,
            "Score_chosen": scores_chosen,
            "Score_rejected": scores_rejected
        })
        

        

st.sidebar.header("Filter")

if data_type == "sft":
    selected_sources = st.sidebar.multiselect(
        "Data Source",
        options=df['Source'].unique(),
        default=df['Source'].unique()
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
        max_value=int(df['Tokens'].max()),
        value=(int(df['Tokens'].min()), int(df['Tokens'].max()))
    )

    filtered_df = df[
        (df['Source'].isin(selected_sources)) &
        (df['Turns'].between(min_turns, max_turns)) &
        (df['Tokens'].between(min_tokens, max_tokens))
    ]
elif data_type == "dpo":
    st.write("筛选的标准是什么？")

st.title("Data Explorer")



PAGE_SIZE = 6
if 'page' not in st.session_state:
    st.session_state.page = 0

total_pages = len(filtered_df) // PAGE_SIZE + (1 if len(filtered_df) % PAGE_SIZE else 0)

def update_page(change):
    st.session_state.page = max(0, min(st.session_state.page + change, total_pages-1))

if data_type == "sft":
    with st.container():

        cols = st.columns([1,1,1])
        with cols[0]: 
            st.markdown(f'''
            <div class="metric-box">
                <div style="font-size:14px;color:#666;">Total</div>
                <div style="font-size:18px;color:#2e86c1;">{len(df):,}</div>
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
elif data_type == "dpo":
    st.write("in progress")
    

st.components.v1.html(f"""
<script>
function updatePage(change) {{
    parent.postMessage({{type: 'session', method: 'callMethod', args: ['update_page', change]}}, '*');
}}
</script>
""")

# 数据导出功能
if st.button("export"):
    # 转换为原始数据格式
    export_data = []
    for _, row in filtered_df.iterrows():
        original_item = next(item for item in data if item['id'] == row['ID'])
        export_data.append(original_item)
    
    buffer = StringIO()
    for item in export_data:
        buffer.write(json.dumps(item) + "\n")
    
    st.download_button(
        label="download",
        data=buffer.getvalue(),
        file_name=f"filtered_data_{uuid.uuid4().hex[:6]}.jsonl",
        mime="application/jsonl+json"
    )

    # 获取各类数据分布
#     turn_num = [len(entry['conversations']) / 2 for entry in data] if st.session_state.show_turn else []
#     source = [entry.get('source', 'unknown') for entry in data] if st.session_state.show_source else []
#     res_source_count = list(Counter(source).items()) if source else []
#     # time_dis = 
#     if st.session_state.show_token:
#         token_num = []
#         conversation = []
#         if st.session_state.chat_template == 'default':
#             token_num = [len(tokenizer.apply_chat_template(entry['conversations'])) for entry in data]
#         else:
#             conv = Conversation(template=TEMPLATES[st.session_state.chat_template])
#             for entry in data:
#                 conv.fill_in_messages(entry)
#                 conversation.extend([msg for role, msg in conv.messages if role == Role.ASSISTANT])
#                 tk_conv = conv.get_tokenized_conversation(tokenizer, 5000)
#                 token_num.append(len(tk_conv['input_ids']))
        
#     else:
#         token_num = []
#         df = pd.DataFrame()

# # 绘制数据统计结果
# st.markdown("<h2 style='text-align: center;'>Statistics</h2>", unsafe_allow_html=True)
# st.markdown(f"""
#     <table style="border-collapse: collapse; width: 50%; margin: auto;">
#         <tr>
#             <td style="padding: 10px; text-align: center; background-color: #e0f7fa; font-size: 18px; font-weight: bold; color: #00796b;">Total data volume</td>
#         </tr>
#         <tr>
#             <td style="padding: 10px; text-align: center; background-color: #ffffff; font-size: 24px; font-weight: bold; color: #333333;">{data_num}</td>
#         </tr>
#     </table>
# """, unsafe_allow_html=True)

# # 动态绘制各图表
# graphs = [
#     (draw_bin_graph, {'data': turn_num, 'num_bins': 10, 'title': 'Turn Distribution', 'x': 'Turn', 'y': 'Count'}, st.session_state.show_turn),
#     (draw_pie_graph, {'data': res_source_count, 'title': 'Source Distribution'}, st.session_state.show_source),
#     (draw_bin_graph, {'data': token_num, 'num_bins': 10, 'title': 'Token Distribution', 'x': 'Token', 'y': 'Count'}, st.session_state.show_token),
#     # (draw_sunburst_graph, {'df': df, 'title': 'Verb-Noun Structure'}, st.session_state.show_vn)
# ]

# columns = st.columns(3)
# curr_col = 0
# for func, params, draw in graphs:
#     if draw:
#         with columns[curr_col]:
#             func(**params)
#         curr_col = (curr_col + 1) % 3
