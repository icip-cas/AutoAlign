"""
Convert alpaca dataset into sharegpt format.

Ref: https://github.com/lm-sys/FastChat/blob/main/fastchat/data/convert_alpaca.py

"""

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str)
    parser.add_argument("--out-file", type=str)
    args = parser.parse_args()

    new_content = []
    with open(args.in_file, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile.readlines()):
            c = json.loads(line)
            q, a = c["instruction"], c["response"]
            new_content.append(
                {
                    "id": f"seed_data_{i}",
                    "conversations": [
                        {"from": "human", "value": q},
                        {"from": "gpt", "value": a},
                    ],
                }
            )

    print(f"#out: {len(new_content)}")
    json.dump(new_content, open(args.out_file, "w"), indent=2, ensure_ascii=False)
