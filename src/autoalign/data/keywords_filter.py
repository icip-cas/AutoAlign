import argparse
import json
import re

from langdetect import detect_langs
from tqdm import tqdm

from autoalign.prompts.harmful import harmful_words
from autoalign.prompts.identity import identity_words

global unwanted_words

def detect_language(text):
    try:
        detected_langs = detect_langs(text)
        lang_code = detected_langs[0].lang
    except Exception:
        lang_code = "unknown"
    return lang_code


def contains_unwanted_words(text):

    for word in unwanted_words:
        if word.lower() in text.lower():
            return True
    return False


def skip(conv, args):

    if args.lang != "all" or args.skip_lang is not None:
        text = "\n".join([x["value"] for x in conv["conversations"]])

        # Check percentage of non-English Unicode characters
        non_eng_chars = sum(1 for c in text if not c.isascii())
        total_chars = len(text)
        if non_eng_chars / total_chars > 0.05:
            return True

        lang_code = detect_language(text)

        if args.lang != "all" and lang_code != args.lang:
            return True

        if lang_code == args.skip_lang:
            return True

    for sentence in conv["conversations"]:
        if contains_unwanted_words(sentence["value"]):
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--filter-keywords", nargs="+", default="identity")
    parser.add_argument("--lang", type=str, default="all", choices=["all", "en"])
    parser.add_argument("--skip-lang", type=str)
    args = parser.parse_args()

    in_file = args.in_file
    lang = args.lang
    skip_lang = args.skip_lang
    assert lang == "all" or skip_lang is None

    unwanted_words = []
    if "identity" in args.filter_keywords:
        unwanted_words.extend(identity_words)
    if "harmful" in args.filter_keywords:
        unwanted_words.extend(harmful_words)

    print(f"unwanted_words: {unwanted_words}")

    # 提取in-file的文件名
    in_file_base_dir = "/".join(in_file.split("/")[:-1])
    in_file_name = in_file.split("/")[-1].split(".")[0]
    in_file_suffix = in_file.split("/")[-1].split(".")[-1]
    out_file = f"{in_file_base_dir}/{in_file_name}_filterd.{in_file_suffix}"
    del_file = f"{in_file_base_dir}/{in_file_name}_deleted.{in_file_suffix}"

    print(f"out_file: {out_file}")

    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    del_content = []
    for conv in tqdm(content):
        if not skip(conv, args):
            new_content.append(conv)
        else:
            del_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2, ensure_ascii=False)
    json.dump(del_content, open(del_file, "w"), indent=2, ensure_ascii=False)
