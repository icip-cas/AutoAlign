import json
from copy import deepcopy


def filter_jsonl_by_lang(input_filepath, target_lang="en"):
    """
    filter the language from input oasst file
    """
    dialog_in_target_lang = []
    with open(input_filepath, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                record = json.loads(line.strip())
                if (
                    "lang" in record["prompt"]
                    and record["prompt"]["lang"] == target_lang
                ):
                    if (
                        "labels" in record["prompt"]
                        and "lang_mismatch" in record["prompt"]["labels"]
                        and record["prompt"]["labels"]["lang_mismatch"]["value"] < 0.2
                    ):  #
                        dialog_in_target_lang.append(record["prompt"])
            except json.JSONDecodeError:
                print("Warn: Skip the line that cannot be parsed as JSON:")
                print(line)
                continue
    return dialog_in_target_lang


def extract_text_and_quality(data):
    """
    extract the text and quality evaluation
    """
    extracted_data = []
    for item in data:
        if "text" in item and "labels" in item:
            if "quality" not in item["labels"]:
                continue
            extracted_data.append(
                {
                    "text": item["text"],
                    "quality": item["labels"]["quality"]["value"],
                    "replies": [],
                }
            )
        if "replies" in item and isinstance(item["replies"], list):
            for replies_item in item["replies"]:
                if "text" in replies_item and "labels" in replies_item:
                    if "quality" not in replies_item["labels"]:
                        continue
                    extracted_data[-1]["replies"].append(
                        {
                            "text": replies_item["text"],
                            "quality": replies_item["labels"]["quality"]["value"],
                        }
                    )
    eft_conv = []
    for item in extracted_data:
        if len(item["replies"]) >= 2:
            eft_conv.append(item)
    return eft_conv


def form_conv_data(conv_data):
    output = []
    index = 0
    with open("data/en_oasst_first_turn.jsonl", "w", encoding="utf-8") as outfile:
        for data in conv_data:
            instruct_text = data["text"]
            replies = data["replies"]
            replies = sorted(replies, key=lambda x: x["quality"])
            item = deepcopy(data)
            item["replies"] = replies
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
            for reply in replies:
                reply_text = reply.get("text", "")
                output.append(
                    {"index": index, "instruct": instruct_text, "response": reply_text}
                )
                index += 1
    return output


if __name__ == "__main__":
    input_file = "data/oasst1/2023-04-12_oasst_all.trees.jsonl"
    target_dialogs = filter_jsonl_by_lang(input_file)
    eft_conv = extract_text_and_quality(target_dialogs)
    output = form_conv_data(eft_conv)
    json.dump(
        output,
        open("data/en_oasst_first_turn.json", "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
