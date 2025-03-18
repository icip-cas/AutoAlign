import streamlit as st
step1 = st.session_state.step1
step2 = st.session_state.step2
step3 = st.session_state.step3
step4 = st.session_state.step4

def modify_lines_containing(original_str, target_str, new_str):
    lines = original_str.split('\n')  # 按换行符分割
    modified = []
    for line in lines:
        if target_str in line:
            # 替换整行内容
            modified.append(new_str)
            # 或仅替换部分内容：line.replace(target_str, new_str)
        else:
            modified.append(line)
    return '\n'.join(modified)

# 示例用法
text = """apple 123
banana 456
cherry 789
apple 999"""

new_text = modify_lines_containing(text, "apple", "FRUIT_REPLACED")
print(new_text)