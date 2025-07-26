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

print("Total instances:", len(data))
print("Average turns:", avg_turn)
print("Average tokens:", avg_token)

# Language statistics
print("\nLanguage distribution statistics:")
total = sum(language_counter.values())
df = pd.DataFrame({
    "language_code": list(language_counter.keys()),
    "n_turns": list(language_counter.values()),
    "percentage": [f"{(count/total)*100:.2f}%" for count in language_counter.values()]
})
print(df.sort_values(by="n_turns", ascending=False).to_string(index=False))