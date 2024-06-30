import json
from argparse import ArgumentParser
from tqdm import tqdm
from copy import deepcopy
import random

parser = ArgumentParser()

parser.add_argument("--input-files", nargs="+", required=True)
parser.add_argument("--chosen-source", type=str)
parser.add_argument("--rejected-source", type=str)
parser.add_argument("--abandon-same-response", action="store_true")
parser.add_argument("--set-source-tag", type=str, default=None) # idx->tag
parser.add_argument("--keep-system-instruction", type=str) # TODO
parser.add_argument("--strategy", type=str)
parser.add_argument("--output-file-path", required=True)

args = parser.parse_args()

set_idx = None
set_tag = None

if args.set_source_tag is not None:
    set_idx, set_tag = args.set_source_tag.split("->")
    set_idx = int(set_idx)

"""
{
    "prompt": "Name two African countries",
    "prompt_id": "5242e4c4142f0f68553e5e6bb16c5105d7a5af2a66a8599f254919cf81b3903b",
    "chosen": [
        {
            "content": "Name two African countries",
            "role": "user"
        },
        {
            "content": "Kenya and Nigeria are two African countries. Kenya is located in East Africa, known for its diverse wildlife and beautiful landscapes, while Nigeria is situated in West Africa and is the continent's most populous country.",
            "role": "assistant"
        }
    ],
    "rejected": [
        {
            "content": "Name two African countries",
            "role": "user"
        },
        {
            "content": "Ethiopia and Morocco.",
            "role": "assistant"
        }
    ],
    "messages": [
        {
            "content": "Name two African countries",
            "role": "user"
        },
        {
            "content": "Kenya and Nigeria are two African countries. Kenya is located in East Africa, known for its diverse wildlife and beautiful landscapes, while Nigeria is situated in West Africa and is the continent's most populous country.",
            "role": "assistant"
        }
    ],
}
"""

""""
策略1：负样本用模型自己生成的，正样本用监督信号
策略2：正样本用两个中间更长的，负样本用两个中间更短的
策略3：正样本用加上principle的，负样本用不加principle的
策略4：正样本用加上principle的，负样本用加负向principle的
"""

def length_strategy(s):
    # set the logest conversation as chosen
    for key in s.keys():
        if key.startswith("conversation_"):
            if "chosen" not in s.keys():
                s["chosen"] = s[key]
            if "rejected" not in s.keys():
                s["rejected"] = s[key]

            if len(s[key][-1]["content"]) > len(s["chosen"][-1]["content"]):
                s["chosen"] = s[key]
            else:
                s["rejected"] = s[key]
    return s

def strategy(preferences_store):

    if args.chosen_source and args.rejected_source:
        for d in preferences_store:
            d["chosen"] = deepcopy(d["conversation_" + args.chosen_source])
            d["rejected"] = deepcopy(d["conversation_" + args.rejected_source])
            d["messages"] = d["chosen"]
            del d["conversation_" + args.chosen_source]
            del d["conversation_" + args.rejected_source]

    elif strategy == "length":
        for d in preferences_store:
            d = length_strategy(d)
            d["messages"] = d["chosen"]

    else:
        raise ValueError()

    return preferences_store

preferences_store = []

pre_data_len = None

for idx, input_file in enumerate(args.input_files):

    print(f"Processing data {idx} {input_file} ...")

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data_len = len(data)
        print(data[0])
        if idx == 0:
            pre_data_len = data_len
        else:
            assert data_len==pre_data_len
        
        if idx == 0:
            for d in tqdm(data):
                # for id contains "chosen" and "rejected"
                d["id"] = d["id"].replace("_chosen", "").replace("_rejected", "")

                for c in d["conversations"]:
                    if c["from"] == "gpt": c["role"] = "assistant"
                    c["content"] = c["value"]
                    del c["value"]
                    del c["from"]
                if idx == set_idx:
                    print(idx)
                    d["source"] = set_tag
                source = d["source"]
                preferences_store.append({
                    "prompt": d["conversations"][0]["content"],
                    "prompt_id": d["id"],
                    f"conversation_{source}": d["conversations"][:2] # only single turn
                })
        else:
            for d, p in tqdm(zip(data, preferences_store)):
                # for id contains "chosen" and "rejected"
                d["id"] = d["id"].replace("_chosen", "").replace("_rejected", "")

                for c in d["conversations"]:
                    if c["from"] == "gpt": c["role"] = "assistant"
                    c["content"] = c["value"]
                    del c["value"]
                    del c["from"]
                if idx == set_idx:
                    d["source"] = set_tag
                source = d["source"]

                if "id" in d and p["prompt_id"] != d["id"]:
                    print(f"Warning: {d['id']} mismatch.")

                p[f"conversation_{source}"] = d["conversations"][:2] # only single turn

preferences_store = strategy(preferences_store)

if args.abandon_same_response:
    num_all_abandon_response = 0
    _preferences_store = []
    for p in preferences_store:
        if p["chosen"][-1]["content"] == p["rejected"][-1]["content"]:
            num_all_abandon_response += 1
        else:
            _preferences_store.append(p)
    preferences_store = _preferences_store
    print(f"Abandon {num_all_abandon_response} data because same response.")

r = random.choice(preferences_store)

print("==============================")

print("Chosen:\n", r["chosen"])

print("------------------------------")

print("Rejected:\n", r["rejected"])

print("==============================")

with open(args.output_file_path, "w", encoding="utf-8") as f:

    for p in preferences_store:
        f.write(
            json.dumps(
                p, 
                ensure_ascii=False
            ) + "\n"
        )