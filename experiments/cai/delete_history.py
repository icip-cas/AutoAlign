import argparse
import random
import json

random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--raw-file", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--source-tag", type=str, required=True)

    args = parser.parse_args()

    input_file = args.input_file
    raw_file = args.raw_file
    output_file = args.output
    source_tag = args.source_tag

    with open(input_file, "r", encoding="utf-8") as f:
        all_data = json.loads(f.read())
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.loads(f.read())
    all_data_dict = {d["id"]: d["conversation"][-1] for d in all_data}
    outputs = []
    for d in raw_data:
        if d["id"] in all_data_dict:
            d["conversation"].append(all_data_dict[d["id"]])
            outputs.append({"id": d["id"], "conversation": d["conversation"], "source": source_tag})
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(outputs, indent=4, ensure_ascii=False))
