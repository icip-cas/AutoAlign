import json
import argparse

def transform_entry(entry):
    # 提取对话中human的内容作为prompt
    human_content = next((conv["value"] for conv in entry["conversations"] if conv["from"] == "human"), "")
    
    # 构建新的数据结构
    return {
        "prompt": human_content,
        "prompt_id": entry["id"],
        "chosen": [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": ""}
        ],
        "rejected": [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": ""}
        ]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cai-file', type=str, required=True, help='输入文件路径')
    args = parser.parse_args()

    # 读取并转换数据
    with open(args.cai_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    transformed_data = [transform_entry(entry) for entry in data]

    # 写回原文件（建议先备份原始文件）
    with open(args.cai_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()