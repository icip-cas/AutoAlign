import json

key_map = {
    "prompt": "prompt",
    "prompt_id": "prompt_id",
    "chosen": "chosen",
    "rejected": "rejected",
    "messages": "conversations",
    "score_chosen": "score_chosen",
    "score_rejected": "score_rejected",
    "role": "from",
    "content": "value",
    "user": "human",
    "assistant": "gpt"
}

def replace_keys(data, key_map):
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            new_key = key_map.get(k, k)
            new_data[new_key] = replace_keys(v, key_map)
        return new_data
    elif isinstance(data, list):
        return [replace_keys(item, key_map) for item in data]
    elif isinstance(data, str):
        return key_map.get(data, data)
    else:
        return data

input_file = "input.json"
output_file = "output.json"


new_data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  
            try:
                data = json.loads(line)
                new_data.append(replace_keys(data, key_map))
            except json.JSONDecodeError as e:
                print(f"JSON 解码错误在行：{line}\n错误：{e}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)
