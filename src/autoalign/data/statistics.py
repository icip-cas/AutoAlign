import argparse
import json
from transformers import AutoTokenizer
from langdetect import detect
from collections import Counter
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--tokenizer-path", type=str, required=True)

args = parser.parse_args()

with open(args.input_file, "r", encoding="utf-8") as f:
    data = json.loads(f.read())

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

avg_turn = 0
avg_token = 0
all_turns = 0
language_counter = Counter()

for d in tqdm(data):
    avg_turn += len(d["conversations"])
    for idx, turn in enumerate(d["conversations"]):
        text = turn["value"]
        avg_token += len(tokenizer(text)["input_ids"])
        all_turns += 1
        
        # 语言检测
        if idx == 0:
            try:
                lang = detect(text)
                language_counter[lang] += 1
            except:
                language_counter["unknown"] += 1

avg_turn /= len(data)
avg_token /= all_turns

print("总实例数: ", len(data))
print("平均轮次: ", avg_turn)
print("平均tokens: ", avg_token)

# 语言统计
print("\n语言分布统计:")
total = sum(language_counter.values())
df = pd.DataFrame({
    "语言代码": list(language_counter.keys()),
    "对话数量": list(language_counter.values()),
    "百分比": [f"{(count/total)*100:.2f}%" for count in language_counter.values()]
})
print(df.sort_values(by="对话数量", ascending=False).to_string(index=False))