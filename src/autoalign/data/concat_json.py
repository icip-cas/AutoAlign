import json
import sys

def concat_json_files(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 根据JSON数据类型进行合并
    if isinstance(data1, list) and isinstance(data2, list):
        # 如果两个都是列表，直接合并列表
        result = data1 + data2
    elif isinstance(data1, dict) and isinstance(data2, dict):
        # 如果两个都是字典，合并字典
        result = {**data1, **data2}
    else:
        # 如果类型不一致，则创建一个包含两者的列表
        result = [data1, data2]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"已成功将 {file1} 和 {file2} 合并到 {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python concat_json.py 文件1.json 文件2.json 输出文件.json")
    else:
        concat_json_files(sys.argv[1], sys.argv[2], sys.argv[3])