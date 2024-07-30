from autoalign.prompts.rlcd import (
    HARMLESS,
    HELPFUL,
    COMPLETENESS,
    INFORMATIVENESS,
    HELPFUL_ZH,
    COMPLETENESS_ZH,
    INFORMATIVENESS_ZH,
)
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
        if "\u4e00" <= char <= "\u9fff":
            return "zh"
        elif "a" <= char.lower() <= "z":
            return "en"
    return "en"


for idx, d in enumerate(all_data):
    ins = d["conversations"][0]["value"]
    system_prompts = (
        en_system_prompts if is_chinese_or_english(ins) == "en" else zh_system_prompts
    )
    system = random.choice(system_prompts)
    if d["conversations"][-1]["from"] == "gpt":
        d["conversations"] = d["conversations"][:-1]
    if d["conversations"][0]["from"] == "system":
        all_chosen_data[idx]["conversations"][0]["value"] = system[0]
        all_rejected_data[idx]["conversation"][0]["value"] = system[1]
    else:
        # insert system message at the first
        all_chosen_data[idx]["conversations"].insert(
            0, {"from": "system", "value": system[0]}
        )
        all_rejected_data[idx]["conversations"].insert(
            0, {"from": "system", "value": system[1]}
        )

with open(args.output_chosen, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_chosen_data, indent=4, ensure_ascii=False))

with open(args.output_rejected, "w", encoding="utf-8") as f:
    f.write(json.dumps(all_rejected_data, indent=4, ensure_ascii=False))
