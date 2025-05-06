#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
import sys

def convert_json_to_jsonl(input_file, output_file=None):
    """
    将JSON文件转换为JSONL格式
    
    参数:
        input_file (str): 输入JSON文件的路径
        output_file (str, optional): 输出JSONL文件的路径，如果不提供，将基于输入文件名自动生成
    
    返回:
        str: 输出文件路径
    """
    # 如果未提供输出文件名，则基于输入文件名自动生成
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.jsonl"
    
    try:
        # 读取JSON文件
        print(f"正在读取JSON文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 将数据写入JSONL文件
        print(f"正在写入JSONL文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in json_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"转换完成! 已处理 {len(json_data)} 条记录")
        return output_file
    
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{input_file}'")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: '{input_file}' 不是有效的JSON文件")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='将JSON文件转换为JSONL格式')
    parser.add_argument('input_file', help='输入JSON文件的路径')
    parser.add_argument('-o', '--output', help='输出JSONL文件的路径 (可选，默认使用输入文件名)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行转换
    output_file = convert_json_to_jsonl(args.input_file, args.output)
    
    print(f"文件已保存至: {output_file}")

if __name__ == "__main__":
    main()