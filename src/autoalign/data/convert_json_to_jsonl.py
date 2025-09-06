#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import os
import sys

def convert_json_to_jsonl(input_file, output_file=None):
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}.jsonl"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in json_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        return output_file
    
    except Exception as e:
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('-o', '--output')
    
    args = parser.parse_args()
    
    output_file = convert_json_to_jsonl(args.input_file, args.output)
    
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    main()