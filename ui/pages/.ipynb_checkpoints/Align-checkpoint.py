import os
import re
import subprocess
import sys
import traceback
import yaml
step1 = os.getenv('step1')
step2 = os.getenv('step2')
step3 = os.getenv('step3')
step4 = os.getenv('step4')
epoch = int(float(os.getenv("epoch")))
eval_path = os.getenv('eval_path')

def replace_input_files(script, epoch):
    # 匹配 --input-files 后的两个参数
    pattern = r'(--input-files\s+)(\S+)(\s+)(\S+)'
    match = re.search(pattern, script)
    
    if match:
        prefix = match.group(1)   # 匹配 --input-files 及其后的空格
        arg1 = match.group(2)     # 第一个参数
        separator = match.group(3)  # 参数间的空白
        arg2 = match.group(4)     # 第二个参数
        
        # 构建新路径
        subdir = f"epoch_{epoch}"
        new_arg1 = os.path.join(arg1, "data_gen", subdir, "chosen_dpo", "chosen.json")
        new_arg2 = os.path.join(arg2, "data_gen", subdir, "rejected_dpo", "rejected.json")
        
        # 创建父目录
        os.makedirs(os.path.dirname(new_arg1), exist_ok=True)
        os.makedirs(os.path.dirname(new_arg2), exist_ok=True)
        
        # 替换参数并保留原始格式
        return re.sub(
            pattern,
            f'{prefix}{new_arg1}{separator}{new_arg2}',
            script
        )
    return script

def rename_parameter(script, old_param, new_param):
    # 匹配参数名及其后的参数值
    pattern = rf'{re.escape(old_param)}(\s+\S+)'
    # 替换参数名并保留参数值
    return re.sub(pattern, f'{new_param}\\1', script)

def modify_lines(script, target_str, new_str):
    pattern = rf'({re.escape(target_str)}\s+)(\S+)'
    # 查找匹配的参数值
    match = re.search(pattern, script)
    
    if match:
        curr_val = match.group(2)
        if os.path.isdir(curr_val):
            new_path = os.path.join(curr_val, new_str)
            os.makedirs(new_path, exist_ok=True)
            # print(f"Generating: {new_path}")
            new_script = re.sub(pattern, rf'\1{new_path}', script)

        else:
            dir, file = os.path.split(curr_val)
            new_path = f"{dir}/{new_str}/{file}"
            new_folder = os.path.join(dir, new_str)
            # print(f"Generating: {new_folder}")
            os.makedirs(new_folder, exist_ok=True)
            new_script = re.sub(pattern, rf'\1{new_path}', script)
        return new_script
    
def appending(script, target_str, new_str):
    pattern = rf'({re.escape(target_str)}\s+)(\S+)'
    # 查找匹配的参数值
    match = re.search(pattern, script)
    
    if match:
        curr_val = match.group(2)
        new_path = os.path.join(curr_val, new_str)
        new_script = re.sub(pattern, rf'\1{new_path}', script)
        return new_script
    
def replacing(script, target_str, new_str):
    pattern = rf'({re.escape(target_str)}\s+)(\S+)'
    # 查找匹配的参数值
    match = re.search(pattern, script)
    if match:
        new_script = re.sub(pattern, rf'\1{new_str}', script)
        return new_script


def get_prev_mod_dir(script):
    target_str = "output_dir"
    pattern = rf'({re.escape(target_str)}\s+)(\S+)'
    match = re.search(pattern, script)
    if match:
        curr_val = match.group(2)
        return curr_val
    
def save_yaml(script, dir):
    data = yaml.safe_load(script)
    filepath = os.path.join(dir, "eval.yaml")
    with open(filepath, 'w') as file:
        yaml.dump(data, file)
    return filepath

def run_scripts(script):
    try:
        subprocess.run(script, text=True, shell=True)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)



try:
    for i in range(epoch):
        subdir = f"epoch_{i+1}"
        prev_subdir = f"epoch_{i}"
        if os.getenv('syn_method') == "MAGPIE":
            step1_1 = modify_lines(step1, "output_folder", subdir)
        elif os.getenv('syn_method') == "Back-Translation":
            step1_1 = appending(step1, "save_filepath", "data_gen")
            step1_1 = modify_lines(step1_1, "save_filepath", subdir)
            step1_1 = appending(step1_1, "save_filepath", "tests.json")
        elif os.getenv('syn_method') == "Self-Instruct":
            step1_1 = appending(step1, "output-path", "data_gen")
            step1_1 = appending(step1_1, "output-path", subdir)

        if os.getenv('method') == "RLCD":# TODO: 这里还有问题，一会回来改
            print("###step2###",step2)
            step2_2 = appending(step2, "input-file", "data_gen/tests.json")
            step2_2 = modify_lines(step2_2, "input-file", subdir)
            step2_2 = appending(step2_2, "output-chosen", "data_gen/chosen.json")
            step2_2 = modify_lines(step2_2, "output-chosen", subdir)
            step2_2 = appending(step2_2, "output-rejected", "data_gen/rejected.json")
            step2_2 = modify_lines(step2_2, "output-rejected", subdir)

            step2_2 = appending(step2_2, "test-file1", "data_gen/chosen.json")
            step2_2 = modify_lines(step2_2, "test-file1", subdir)
            step2_2 = appending(step2_2, "output-dir1", "data_gen")
            step2_2 = modify_lines(step2_2, "output-dir1", subdir)
            step2_2 = appending(step2_2, "output-dir1", "chosen_dpo")
            step2_2 = rename_parameter(step2_2, "test-file1", "test-file")
            step2_2 = rename_parameter(step2_2, "output-dir1", "output-dir")

            step2_2 = appending(step2_2, "test-file2", "data_gen/rejected.json")
            step2_2 = modify_lines(step2_2, "test-file2", subdir)
            step2_2 = appending(step2_2, "output-dir2", "data_gen")
            step2_2 = modify_lines(step2_2, "output-dir2", subdir)
            step2_2 = appending(step2_2, "output-dir2", "rejected_dpo")
            step2_2 = rename_parameter(step2_2, "test-file2", "test-file")
            step2_2 = rename_parameter(step2_2, "output-dir2", "output-dir")

            step2_2 = replace_input_files(step2_2, i+1)
            step2_2 = appending(step2_2, "output-file-path", "sample_ans")
            step2_2 = modify_lines(step2_2, "output-file-path", subdir)
            step2_2 = appending(step2_2, "output-file-path", "preference_data.json")
            print("###step2_2###",step2_2)
        elif os.getenv('method') == "SPIN":
            step2_2 = modify_lines(step2, "output-file-path", subdir)
        elif os.getenv('method') == "Self_Rewarding":
            step2_2 = appending(step2, "instruction-path", "data_gen/tests.json")
            step2_2 = modify_lines(step2_2, "instruction-path", subdir)
            step2_2 = appending(step2_2, "output-path", "sample_ans")
            step2_2 = modify_lines(step2_2, "output-path", subdir)
        elif os.getenv('method') == "CAI_sft":# TODO: 未测试，效果不明
            step2_2 = 
            step2_2 = appending(step2_2, "input-file", "data_gen/tests.json")
            step2_2 = modify_lines(step2_2, "input-file", subdir)
            step2_2 = appending(step2_2, "output-sft", "sample_ans")
            step2_2 = modify_lines(step2_2, "output-sft", subdir)
            step2_2 = appending(step2_2, "output-sft", "preference_data.json")
        elif os.getenv('method') == "CAI_dpo":# TODO: 未测试实际效果
            step2_2 = appending(step2, "input-file", "data_gen/tests.json")
            step2_2 = modify_lines(step2_2, "input-file", subdir)
            step2_2 = appending(step2_2, "output-file", "sample_ans")
            step2_2 = modify_lines(step2_2, "output-file", subdir)
            step2_2 = appending(step2_2, "output-file", "preference_data.json")
        
        step3_3 = appending(step3, "data_path", "sample_ans")
        if i>0:
            print(f"prev-path: {prev_path}")
            step3_3 = replacing(step3_3, "model_name_or_path", prev_path)
        step3_3 = appending(step3_3, "data_path", "preference_data.json")
        step3_3 = modify_lines(step3_3, "data_path", subdir)
        step3_3 = modify_lines(step3_3, "output_dir", "model")
        step3_3 = modify_lines(step3_3, "output_dir", subdir)
        prev_path = get_prev_mod_dir(step3_3)

        step4_4 = appending(step4, "model_path:", "model")
        step4_4 = modify_lines(step4_4, "model_path:",subdir)
        eval_path_new = os.path.join(os.path.join(eval_path, "eval"), subdir)
        os.makedirs(eval_path_new, exist_ok=True)
        eval_filepath = save_yaml(step4_4, eval_path_new)
        if epoch == 1 or i+1 == epoch:
            step4_41 = f"""
autoalign-cli eval --config-path {eval_filepath} --force 2>&1 | tee outputs/eval.log
"""
#         if epoch == 1 or i+1 == epoch:
#             step4_41 = f"""
# echo "Finished!" | tee outputs/eval.log
# """
        else:
            step4_41 = f"""
autoalign-cli eval --config-path {eval_filepath} --force 2>&1 | tee outputs/eval.log; echo "###page5###" >> outputs/eval.log
"""
#         else:
#             step4_41 = f"""
# echo "Working!" | tee outputs/eval.log; echo "###page5###" >> outputs/eval.log
# """
        print(f"第{i+1}轮：")
        print(f"第1部分：{step1_1}")
        print(f"第2部分：{step2_2}")
        print(f"第3部分：{step3_3}")
        print(f"第4部分：{step4_41}")
        run_scripts(step1_1)
        run_scripts(step2_2)
        run_scripts(step3_3)
        run_scripts(step4_41)

        

        
except Exception as e:
    traceback.print_exc()
    sys.exit(1)