from zhuque.prompts.rlcd import HARMLESS, HELPFUL, COMPLETENESS, INFORMATIVENESS, HARMLESS_ZH, HELPFUL_ZH, COMPLETENESS_ZH, INFORMATIVENESS_ZH
import argparse
import random
import json
from copy import deepcopy

random.seed(42)

parser = argparse.ArgumentParser()

parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-chosen", type=str, required=True)
parser.add_argument("--output-rejected", type=str, required=True)

args = parser.parse_args()

en_system_prompts = HARMLESS + HELPFUL + COMPLETENESS + INFORMATIVENESS
zh_system_prompts = HELPFUL_ZH + HELPFUL_ZH + COMPLETENESS_ZH + INFORMATIVENESS_ZH

with open(args.input_file, "r", encoding="utf-8") as f:
    all_data = json.loads(f.read())

all_chosen_data = deepcopy(all_data)
all_rejected_data = deepcopy(all_data)

def is_chinese_or_english(string):
    for char in string:
        if '\u4e00' <= char <= '\u9fff':
            return "zh"
        elif 'a' <= char.lower() <= 'z':
            return "en"
    return "en"
        
for idx, d in enumerate(all_data):
    ins = d["conversations"][0]["value"]
    system_prompts = en_system_prompts if is_chinese_or_english(ins) == "en" else zh_system_prompts
    system = random.choice(system_prompts)
    all_chosen_data[idx]["system"] = system[0]
    all_rejected_data[idx]["system"] = system[1]

with open(args.output_chosen, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_chosen_data, ensure_ascii=False))

with open(args.output_rejected, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_rejected_data, ensure_ascii=False))