import argparse
import json

def transform_data(data):
    """执行数据格式转换"""
    return [
        {
            "id": item["id"],
            "conversations": [
                {"role": conv["from"], "content": conv["value"]}
                for conv in item["conversations"]
            ]
        }
        for item in data
    ]

def process_file(file_path):
    """直接覆盖原文件的实现"""
    try:
        # 读取并转换数据
        with open(file_path, 'r', encoding='utf-8') as f:
            transformed = transform_data(json.load(f))
        
        # 写入原文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(transformed, f, indent=2, ensure_ascii=False)
        
        print(f"文件已更新: {file_path}")

    except FileNotFoundError:
        print(f"错误：文件不存在 - {file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON解析错误：{e}")
    except Exception as e:
        print(f"操作失败：{str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='文件原地转换工具')
    parser.add_argument('--file', required=True, help='目标文件路径')
    args = parser.parse_args()
    process_file(args.file)