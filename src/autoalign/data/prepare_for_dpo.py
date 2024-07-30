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
parser.add_argument("--set-source-tag", type=str, default=None)  # idx->tag
parser.add_argument("--remove-system-message", action="store_true")
parser.add_argument("--strategy", type=str)
parser.add_argument("--output-file-path", required=True)

args = parser.parse_args()

if args.chosen_source and args.rejected_source:
    assert args.chosen_source != args.rejected_source

set_idx = None
set_tag = None

if args.set_source_tag is not None:
    set_idx, set_tag = args.set_source_tag.split("->")
    set_idx = int(set_idx)


def length_strategy(d):
    # set the logest conversation as chosen
    for key in d.keys():
        if key.startswith("conversation_"):
            if "chosen" not in d.keys():
                d["chosen"] = d[key]
            if "rejected" not in d.keys():
                d["rejected"] = d[key]

            if len(d[key][-1]["content"]) > len(d["chosen"][-1]["content"]):
                d["chosen"] = d[key]
            else:
                d["rejected"] = d[key]
    d["conversations"] = d["chosen"]
    return d


def given_strategy(d):

    d["chosen"] = deepcopy(d["conversation_" + args.chosen_source])
    d["rejected"] = deepcopy(d["conversation_" + args.rejected_source])
    del d["conversation_" + args.chosen_source]
    del d["conversation_" + args.rejected_source]
    return d


def strategy(preferences_store):

    if args.chosen_source and args.rejected_source:
        for d in preferences_store:
            d = given_strategy(d)

    elif strategy == "length":
        for d in preferences_store:
            d = length_strategy(d)

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
            assert data_len == pre_data_len

        if idx == 0:
            for d in tqdm(data):
                # for id contains "chosen" and "rejected"
                d["id"] = d["id"].replace("_chosen", "").replace("_rejected", "")

                if idx == set_idx:
                    print(idx)
                    d["source"] = set_tag
                source = d["source"]

                if d["conversations"][0]["from"] == "system":
                    if args.remove_system_message:
                        d["conversations"] = d["conversations"][1:]
                        prompt = d["conversations"][0]["value"]
                    else:
                        prompt = d["conversations"][1]["value"]
                else:
                    prompt = d["conversations"][0]["value"]

                preferences_store.append(
                    {
                        "prompt": prompt,
                        "prompt_id": d["id"],
                        f"conversation_{source}": d["conversations"],
                    }
                )
        else:
            for d, p in tqdm(zip(data, preferences_store)):
                # for id contains "chosen" and "rejected"
                if "id" in d:
                    d["id"] = d["id"].replace("_chosen", "").replace("_rejected", "")

                if idx == set_idx:
                    d["source"] = set_tag
                source = d["source"]

                if (
                    d["conversations"][0]["from"] == "system"
                    and args.remove_system_message
                ):
                    d["conversations"] = d["conversations"][1:]

                if "id" in d and p["prompt_id"] != d["id"]:
                    print(f"Warning: {d['id']} mismatch.")

                p[f"conversation_{source}"] = d["conversations"]

preferences_store = strategy(preferences_store)

if args.abandon_same_response:
    num_all_abandon_response = 0
    _preferences_store = []
    for p in preferences_store:
        if p["chosen"][-1]["value"] == p["rejected"][-1]["value"]:
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

    f.write(json.dumps(preferences_store, indent=4, ensure_ascii=False))
