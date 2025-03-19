import subprocess
import os

trun = 3  # 轮次
input_file = "input.txt"
output_file = "output.txt"
model_name = "gpt-3"
max_length = 100
device = "cuda"
batch_size = 8
num_return_sequences = 1
num_beams = 5
num_beam_groups = 1
diversity_penalty = 0.0
temperature = 1.0
top_k = 50
top_p = 1.0
repetition_penalty = 1.0
length_penalty = 1.0
no_repeat_ngram_size = 0
stop_token = None
seed = 42
do_sample = True
num_samples = 1
max_turn = 3

# 前端传入的模型保存目录
model_save_dir = "/path/to/model/save/directory"

# 前端传入的数据保存目录
data_save_dir = "/path/to/data/save/directory"

# 初始化输入模型路径
input_model_path = model_save_dir

for i in range(trun):
    print(f"正在执行第 {i + 1} 轮任务...")
    
    # 生成当前轮的模型保存路径
    output_model_path = os.path.join(model_save_dir, f"model-{i + 1}")
    
    # 生成当前轮的数据保存路径
    current_data_save_path = os.path.join(data_save_dir, f"data-{i + 1}")
    
    # 如果数据保存路径不存在，则创建
    os.makedirs(current_data_save_path, exist_ok=True)
    
    # 构建命令
    command = [
        "sh", "single_turn.sh",
        "--input_file", input_file,
        "--output_file", output_file,
        "--model_name", model_name,
        "--max_length", str(max_length),
        "--device", device,
        "--batch_size", str(batch_size),
        "--num_return_sequences", str(num_return_sequences),
        "--num_beams", str(num_beams),
        "--num_beam_groups", str(num_beam_groups),
        "--diversity_penalty", str(diversity_penalty),
        "--temperature", str(temperature),
        "--top_k", str(top_k),
        "--top_p", str(top_p),
        "--repetition_penalty", str(repetition_penalty),
        "--length_penalty", str(length_penalty),
        "--no_repeat_ngram_size", str(no_repeat_ngram_size),
        "--stop_token", str(stop_token) if stop_token else "None",
        "--seed", str(seed),
        "--do_sample", str(do_sample).lower(),
        "--num_samples", str(num_samples),
        "--max_turn", str(max_turn),
        "--turn", str(i),
        "--input_model_path", input_model_path,
        "--output_model_path", output_model_path,
        "--data_save_path", current_data_save_path  # 添加当前轮的数据保存路径
    ]

    # 执行命令
    result = subprocess.run(command, capture_output=True, text=True)
    
    # 将当前轮的输出模型路径作为下一轮的输入模型路径
    input_model_path = output_model_path

print("over")