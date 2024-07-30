import argparse
import json
from transformers import AutoTokenizer

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

for d in data:
    avg_turn += len(d["conversations"])
    for turn in d["conversations"]:
        avg_token += len(tokenizer(turn["value"])["input_ids"])
        all_turns += 1

avg_turn /= len(data)
avg_token /= all_turns

print("Total instances: ", len(data))
print("Average turns: ", avg_turn)
print("Average tokens: ", avg_token)
