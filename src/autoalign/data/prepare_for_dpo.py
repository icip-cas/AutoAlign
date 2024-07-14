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
[
    {
      "prompt":"part 1. definition\ngiven a story, answer the question about the story. the question is the last sentence in the input. these stories can be difficult due to their length and how each story has at least one of the three following scenarios: the first is when the individual's belief matches reality, the second is when the individual's belief does not match reality, and the third is when an individual has a false belief about another individual's beliefs. the question will ask about the location of an object in the story with respect to either none or one of the three scenarios.\npart 2. example\njacob entered the dining_room. william entered the dining_room. the tomato is in the green_drawer. william exited the dining_room. jacob moved the tomato to the blue_cupboard. jacob is in the dining_room. olivia entered the dining_room. the cucumber is in the blue_cupboard. olivia exited the dining_room. jacob moved the cucumber to the green_drawer. william entered the pantry. jacob entered the pantry. the asparagus is in the red_cupboard. jacob exited the pantry. william moved the asparagus to the green_pantry. abigail entered the hall. william entered the hall. the persimmon is in the blue_pantry. william exited the hall. abigail moved the persimmon to the blue_envelope. where does abigail think that william searches for the persimmon?\nanswer: blue_pantry\nexplanation: the persimmon was last in the blue_pantry before william exited the hall. after william exited the hall, abigail moved the persimmon to the blue_envelope, so she knows where william will look for it.\npart 3. exercise\nethan entered the office. charlotte entered the office. the lettuce is in the blue_box. charlotte exited the office. ethan moved the lettuce to the red_treasure_chest. ethan exited the office. charlotte entered the office. where is the lettuce really? isabella entered the sunroom. charlotte entered the sunroom. the green_pepper is in the green_drawer. charlotte exited the sunroom. isabella moved the green_pepper to the red_pantry. isabella exited the sunroom. charlotte entered the sunroom. where does isabella think that charlotte searches for the green_pepper? charlotte entered the lounge. ethan entered the lounge. the beans is in the green_cupboard. ethan exited the lounge. charlotte moved the beans to the blue_crate. where is the beans really? ella entered the lounge. charlotte is in the lounge. the broccoli is in the blue_crate. charlotte exited the lounge. ella moved the broccoli to the green_cupboard. ella exited the lounge. charlotte entered the lounge. where is the broccoli really?\nanswer:",
      "prompt_id":"58bde641397654a889a019dd74956009ab8a5fd1a62e7d3d100e42a4075c8f6c",
      "chosen":[
        {
          "value":"part 1. definition\ngiven a story, answer the question about the story. the question is the last sentence in the input. these stories can be difficult due to their length and how each story has at least one of the three following scenarios: the first is when the individual's belief matches reality, the second is when the individual's belief does not match reality, and the third is when an individual has a false belief about another individual's beliefs. the question will ask about the location of an object in the story with respect to either none or one of the three scenarios.\npart 2. example\njacob entered the dining_room. william entered the dining_room. the tomato is in the green_drawer. william exited the dining_room. jacob moved the tomato to the blue_cupboard. jacob is in the dining_room. olivia entered the dining_room. the cucumber is in the blue_cupboard. olivia exited the dining_room. jacob moved the cucumber to the green_drawer. william entered the pantry. jacob entered the pantry. the asparagus is in the red_cupboard. jacob exited the pantry. william moved the asparagus to the green_pantry. abigail entered the hall. william entered the hall. the persimmon is in the blue_pantry. william exited the hall. abigail moved the persimmon to the blue_envelope. where does abigail think that william searches for the persimmon?\nanswer: blue_pantry\nexplanation: the persimmon was last in the blue_pantry before william exited the hall. after william exited the hall, abigail moved the persimmon to the blue_envelope, so she knows where william will look for it.\npart 3. exercise\nethan entered the office. charlotte entered the office. the lettuce is in the blue_box. charlotte exited the office. ethan moved the lettuce to the red_treasure_chest. ethan exited the office. charlotte entered the office. where is the lettuce really? isabella entered the sunroom. charlotte entered the sunroom. the green_pepper is in the green_drawer. charlotte exited the sunroom. isabella moved the green_pepper to the red_pantry. isabella exited the sunroom. charlotte entered the sunroom. where does isabella think that charlotte searches for the green_pepper? charlotte entered the lounge. ethan entered the lounge. the beans is in the green_cupboard. ethan exited the lounge. charlotte moved the beans to the blue_crate. where is the beans really? ella entered the lounge. charlotte is in the lounge. the broccoli is in the blue_crate. charlotte exited the lounge. ella moved the broccoli to the green_cupboard. ella exited the lounge. charlotte entered the lounge. where is the broccoli really?\nanswer:",
          "from":"human"
        },
        {
          "value":"1. red_treasure_chest\nexplanation: the lettuce was last moved to the red_treasure_chest by ethan.\n\n2. green_drawer\nexplanation: isabella moved the green_pepper to the red_pantry after charlotte exited the sunroom. charlotte didn't see this change, so she would still think the green_pepper is in the green_drawer.\n\n3. blue_crate\nexplanation: charlotte moved the beans to the blue_crate, which is their current location.\n\n4. green_cupboard\nexplanation: ella moved the broccoli to the green_cupboard after charlotte exited the lounge, making it the real location of the broccoli.",
          "from":"gpt"
        }
      ],
      "rejected":[
        {
          "value":"part 1. definition\ngiven a story, answer the question about the story. the question is the last sentence in the input. these stories can be difficult due to their length and how each story has at least one of the three following scenarios: the first is when the individual's belief matches reality, the second is when the individual's belief does not match reality, and the third is when an individual has a false belief about another individual's beliefs. the question will ask about the location of an object in the story with respect to either none or one of the three scenarios.\npart 2. example\njacob entered the dining_room. william entered the dining_room. the tomato is in the green_drawer. william exited the dining_room. jacob moved the tomato to the blue_cupboard. jacob is in the dining_room. olivia entered the dining_room. the cucumber is in the blue_cupboard. olivia exited the dining_room. jacob moved the cucumber to the green_drawer. william entered the pantry. jacob entered the pantry. the asparagus is in the red_cupboard. jacob exited the pantry. william moved the asparagus to the green_pantry. abigail entered the hall. william entered the hall. the persimmon is in the blue_pantry. william exited the hall. abigail moved the persimmon to the blue_envelope. where does abigail think that william searches for the persimmon?\nanswer: blue_pantry\nexplanation: the persimmon was last in the blue_pantry before william exited the hall. after william exited the hall, abigail moved the persimmon to the blue_envelope, so she knows where william will look for it.\npart 3. exercise\nethan entered the office. charlotte entered the office. the lettuce is in the blue_box. charlotte exited the office. ethan moved the lettuce to the red_treasure_chest. ethan exited the office. charlotte entered the office. where is the lettuce really? isabella entered the sunroom. charlotte entered the sunroom. the green_pepper is in the green_drawer. charlotte exited the sunroom. isabella moved the green_pepper to the red_pantry. isabella exited the sunroom. charlotte entered the sunroom. where does isabella think that charlotte searches for the green_pepper? charlotte entered the lounge. ethan entered the lounge. the beans is in the green_cupboard. ethan exited the lounge. charlotte moved the beans to the blue_crate. where is the beans really? ella entered the lounge. charlotte is in the lounge. the broccoli is in the blue_crate. charlotte exited the lounge. ella moved the broccoli to the green_cupboard. ella exited the lounge. charlotte entered the lounge. where is the broccoli really?\nanswer:",
          "from":"human"
        },
        {
          "value":"of course! i'll do my best to assist you with your questions. please go ahead and ask your first question. 😊",
          "from":"gpt"
        }
      ],
      "conversations":[
        {
          "value":"part 1. definition\ngiven a story, answer the question about the story. the question is the last sentence in the input. these stories can be difficult due to their length and how each story has at least one of the three following scenarios: the first is when the individual's belief matches reality, the second is when the individual's belief does not match reality, and the third is when an individual has a false belief about another individual's beliefs. the question will ask about the location of an object in the story with respect to either none or one of the three scenarios.\npart 2. example\njacob entered the dining_room. william entered the dining_room. the tomato is in the green_drawer. william exited the dining_room. jacob moved the tomato to the blue_cupboard. jacob is in the dining_room. olivia entered the dining_room. the cucumber is in the blue_cupboard. olivia exited the dining_room. jacob moved the cucumber to the green_drawer. william entered the pantry. jacob entered the pantry. the asparagus is in the red_cupboard. jacob exited the pantry. william moved the asparagus to the green_pantry. abigail entered the hall. william entered the hall. the persimmon is in the blue_pantry. william exited the hall. abigail moved the persimmon to the blue_envelope. where does abigail think that william searches for the persimmon?\nanswer: blue_pantry\nexplanation: the persimmon was last in the blue_pantry before william exited the hall. after william exited the hall, abigail moved the persimmon to the blue_envelope, so she knows where william will look for it.\npart 3. exercise\nethan entered the office. charlotte entered the office. the lettuce is in the blue_box. charlotte exited the office. ethan moved the lettuce to the red_treasure_chest. ethan exited the office. charlotte entered the office. where is the lettuce really? isabella entered the sunroom. charlotte entered the sunroom. the green_pepper is in the green_drawer. charlotte exited the sunroom. isabella moved the green_pepper to the red_pantry. isabella exited the sunroom. charlotte entered the sunroom. where does isabella think that charlotte searches for the green_pepper? charlotte entered the lounge. ethan entered the lounge. the beans is in the green_cupboard. ethan exited the lounge. charlotte moved the beans to the blue_crate. where is the beans really? ella entered the lounge. charlotte is in the lounge. the broccoli is in the blue_crate. charlotte exited the lounge. ella moved the broccoli to the green_cupboard. ella exited the lounge. charlotte entered the lounge. where is the broccoli really?\nanswer:",
          "from":"human"
        },
        {
          "value":"1. red_treasure_chest\nexplanation: the lettuce was last moved to the red_treasure_chest by ethan.\n\n2. green_drawer\nexplanation: isabella moved the green_pepper to the red_pantry after charlotte exited the sunroom. charlotte didn't see this change, so she would still think the green_pepper is in the green_drawer.\n\n3. blue_crate\nexplanation: charlotte moved the beans to the blue_crate, which is their current location.\n\n4. green_cupboard\nexplanation: ella moved the broccoli to the green_cupboard after charlotte exited the lounge, making it the real location of the broccoli.",
          "from":"gpt"
        }
      ],
      "score_chosen":8.5,
      "score_rejected":1.0
    }
]
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
            d["conversations"] = d["chosen"]
            del d["conversation_" + args.chosen_source]
            del d["conversation_" + args.rejected_source]

    elif strategy == "length":
        for d in preferences_store:
            d = length_strategy(d)
            d["conversations"] = d["chosen"]

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
                if "id" in d:
                    d["id"] = d["id"].replace("_chosen", "").replace("_rejected", "")

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