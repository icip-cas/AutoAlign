from autoalign.prompts.constitutions import constitutions, history_stage1, history_stage2, history_stage3
import argparse
import random
import json
from copy import deepcopy

random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--last-stage-output", type=str)

    args = parser.parse_args()

    input_file = args.input_file
    stage = args.stage
    output_file = args.output
    last_stage_output = args.last_stage_output

    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.loads(f.read())

    if last_stage_output is not None and stage > 1:
        with open(last_stage_output, "r", encoding="utf-8") as f:
            last_stage_data = json.loads(f.read())
        last_stage_dict = {d["id"]: d["conversation"][-(2 * stage - 3):] for d in last_stage_data}
        for d in all_data:
            if d["id"] in last_stage_dict:
                d["conversations"] = d["conversations"] + last_stage_dict[d["id"]]

    # assign constitution
    if stage == 0:
        for d in all_data:
            d["constitution"] = random.choice(constitutions)
        if "id" not in all_data[0]:
            for idx, d in enumerate(all_data):
                d["id"] = idx
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(all_data, indent=4, ensure_ascii=False))
    else:
        histories = [[], history_stage1, history_stage2, history_stage3]

        for idx, d in enumerate(all_data):
            constitution = d["constitution"]

            if all_data[idx]["conversations"][0]["from"] == "system":
                all_data[idx]["conversations"] = (
                    all_data[idx]["conversations"][0] + deepcopy(histories[stage]) + all_data[idx]["conversations"][1:]
                )
            else:
                all_data[idx]["conversation"] = deepcopy(histories[stage]) + all_data[idx]["conversation"]

            if stage == 2:
                all_data[idx]["conversations"].append({"from": "human", "value": constitution["critic"]})
            elif stage == 3:
                all_data[idx]["conversations"].append({"from": "human", "value": constitution["revision"]})
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(all_data, indent=4, ensure_ascii=False))
