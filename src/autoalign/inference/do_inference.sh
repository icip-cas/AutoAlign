export model_path=$1
export prompt_path=$2
export save_path=$3
export batch_size=$4
export per_gpu=$5
# echo 'Please provide model_path, prompt_path, save_path, batch_size, per_gpu as arguments.
# For example: bash do_inference.sh /path/llama3-8b-instruct /home/prompt.json /home/save_file.json 32 1"

# Model_path: the path to the directory where the model is saved.
# Prompt_path: the path to the prompt file which contains the prompts to be aligned.
# Save_path: the path to the directory where the aligned prompts will be saved.
# Batch_size: the batch size for inference. If not provided, the default batch size is 32.
# Per_gpu: the number of gpus every 

# prompt file should format as json file, each line is a json object with the following format:
# {
#     "prompt": "prompt text",
#     "id": "prompt id"
# }
# '

echo "
请提供 model_path, prompt_path, save_path, batch_size, per_gpu 作为参数。
例如：
bash do_inference.sh /path/llama3-8b-instruct /home/prompt.json /home/save_file.json 32 1

解释：
- Model_path: 模型保存目录的路径。
- Prompt_path: 包含对齐提示的提示文件路径。
- Save_path: 对齐提示保存目录的路径。
- Batch_size: 推理的批量大小。如果未提供，默认批量大小为32。
- Per_gpu: 每个模型占用GPU的数量，如per_gpu=1，共有8个GPU，则每个模型占用一个GPU，启动8线程进行推理。

提示文件应该格式化为JSON文件，每一行都是一个包含以下格式的JSON对象：
json
{
    "prompt": "prompt text",
    "id": "prompt id"
}

请在generate_config.json中配置生成参数。
"

if [ -z "$prompt_path" ] || [ -z "$save_path" ]; then
    echo "Please provide prompt_path and save_path."
    exit 1
fi

if [ -z "$batch_size" ]; then
    batch_size=32
fi

echo "Prompt path: $prompt_path"
echo "Save path: $save_path"
echo "Batch size: $batch_size, if not provided, default batch size is 32"

python inference.py --model_path $model_path --prompt_path $prompt_path --save_path $save_path --batch_size $batch_size --per_gpu $per_gpu