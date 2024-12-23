import json
import random
import argparse
import os
import sys
from math import ceil
from tqdm import tqdm
from copy import deepcopy
from autoalign.prompts.judge import META_JUDGE_EN

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dpo_dataset_generator import DPODatasetGenerator
from utils import load_jsonlines, load_json, dump_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--template-name",
        type=str,
        required=True,
        help="The template name.",
    )
    parser.add_argument(
        "--seed-data-path",
        type=str,
        required=True,
        help="The path to the the seed data. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        required=True,
        help="The optimized sft baseline.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    args = parser.parse_args()
    return args


def process_scores(golden: list, score_data: list):
    index = 0
    raw_eft_data = []
    for data in golden:
        instruct_text = data["text"]
        replies = data["replies"]
        replies = sorted(replies, key=lambda x: x["quality"])
        # outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
        scored_replies = []
        for conv_data in score_data[index:]:
            if conv_data["instruct"] == data["text"]:
                # processing score
                scores = [
                    s["output"]
                    for s in conv_data["score"]["choices"]
                    if s["output"] is not None
                ]
                avg_score = -1
                if len(scores) > 0:
                    avg_score = sum(scores) / len(scores)
                conv_data["score"]["choices"]
                scored_replies.append(
                    {
                        "index": conv_data["index"],
                        "text": conv_data["response"],
                        "quality": avg_score,
                    }
                )
                index += 1
            else:
                break
        sorted_scored_replies = sorted(scored_replies, key=lambda x: x["quality"])
        assert len(sorted_scored_replies) == len(replies)
        for idx, item in enumerate(sorted_scored_replies):
            rank = index - len(sorted_scored_replies) + idx
            if item["index"] == rank:
                assert replies[idx]["text"] == item["text"]
                raw_eft_data.append(
                    {
                        "id": str(len(raw_eft_data)),
                        "conversations": [
                            {
                                "from": "human",
                                "value": META_JUDGE_EN.format(
                                    instruction=instruct_text, response=item["text"]
                                ),
                            },
                            {
                                "from": "gpt",
                                "value": "Score: {reward}".format(
                                    reward=replies[idx]["quality"] * 5
                                ),
                            },
                        ],
                    }
                )
    return raw_eft_data


def filter_seed_data(raw_eft_data, seed_data):
    filtered_eft_data = []
    for eft_data in tqdm(raw_eft_data):
        found = 0
        for seed in seed_data:
            seed_conv = META_JUDGE_EN.format(
                instruction=seed["conversations"][0]["value"],
                response=seed["conversations"][1]["value"],
            )
            if eft_data["conversations"][0]["value"] == seed_conv:
                found = 1
                break
        if found == 0:
            filtered_eft_data.append(eft_data)

    return filtered_eft_data


def balance_split(filtered_eft_data):
    print(filtered_eft_data)
    rewards = {0.0: 0, 1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
    threshold = deepcopy(rewards)
    train_set = deepcopy(rewards)
    train_eft_data = []
    eval_eft_data = []
    for eft_data in tqdm(filtered_eft_data):
        reward = int(
            float(eft_data["conversations"][1]["value"].replace("Score: ", ""))
        )
        threshold[float(reward)] += 1

    over_avg_num = 0
    sum_of_limit = 2161
    for value in threshold.values():
        if value <= sum_of_limit // len(threshold.values()):
            sum_of_limit -= value
        else:
            over_avg_num += 1
    if over_avg_num > 0:
        for key in threshold.keys():
            threshold[key] = min(ceil(sum_of_limit / over_avg_num), threshold[key])
            train_set[key] = round(threshold[key] * 1630 / 2161)
            assert train_set[key] < threshold[key]

    random.shuffle(filtered_eft_data)
    for eft_data in tqdm(filtered_eft_data):
        reward = int(
            float(eft_data["conversations"][1]["value"].replace("Score: ", ""))
        )
        rewards[float(reward)] += 1
        eft_data["conversations"][1]["value"] = "Score: {}".format(str(reward))
        if rewards[reward] > threshold[reward]:
            continue
        if rewards[reward] > train_set[reward]:
            eval_eft_data.append(eft_data)
        else:
            train_eft_data.append(eft_data)
    return train_eft_data, eval_eft_data


if __name__ == "__main__":
    args = parse_args()
    golden = load_jsonlines("data/en_oasst_first_turn.jsonl")

    generator = DPODatasetGenerator(
        model_name=args.model_path,
        instruct_generator=None,
        ift_dataset=args.seed_data_path,
        template_name=args.template_name,
        backend=args.backend,
        num_gpus_per_model=args.num_gpus_per_model,
    )

    formed_data = load_json("data/en_oasst_first_turn.json")
    score_data = generator.generate_scores(instr_resp_pairs=formed_data)

    raw_eft_data = process_scores(golden, score_data)
    seed_data = json.load(open(args.seed_data_path, "r"))
    filtered_eft_data = filter_seed_data(raw_eft_data, seed_data)

    train_eft_data, eval_eft_data = balance_split(filtered_eft_data)

    dump_json(
        sorted(train_eft_data, key=lambda x: int(x["id"])),
        "data/{}/train_eft_data.json".format(args.model_id),
        indent=2,
    )

    dump_json(
        sorted(eval_eft_data, key=lambda x: int(x["id"])),
        "data/{}/eval_eft_data.json".format(args.model_id),
        indent=2,
    )
