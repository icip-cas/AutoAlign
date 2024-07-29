import argparse
import json
import re

from langdetect import detect_langs
from tqdm import tqdm

from autoalign.prompts.harmful import unwanted_words

print(unwanted_words)

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
        if non_eng_chars / total_chars > .05:
            return True

        lang_code = detect_language(text)

        if args.lang != "all" and lang_code != args.lang:
            return True

        if lang_code == args.skip_lang:
            return True

    if args.reduce_rep:
        for sentence in conv["conversations"]:
            val = sentence["value"]
            sub = re.search(r"(\d)\1{8}", val)
            if sub is not None:
                return True

    for sentence in conv["conversations"]:
        if contains_unwanted_words(sentence["value"]):
            return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="")
    parser.add_argument("--lang", type=str, default="all",
                        choices=["all", "en"])
    parser.add_argument("--skip-lang", type=str)
    parser.add_argument("--reduce-rep", action="store_true")
    args = parser.parse_args()

    in_file = args.in_file
    out_file = args.out_file
    lang = args.lang
    skip_lang = args.skip_lang
    reduce_rep = args.reduce_rep
    assert (lang == "all" or skip_lang is None)

    if out_file == "":
        out_file = "sharegpt_clean"
        if lang != "all":
            out_file += "_" + lang
        if skip_lang is not None:
            out_file += "_skip_" + skip_lang
        if reduce_rep:
            out_file += "_reduce_rep"
        out_file += ".json"

    content = json.load(open(in_file, "r"))
    num_conv = len(content)

    new_content = []
    for conv in tqdm(content):
        if not skip(conv, args):
            new_content.append(conv)

    print(f"return {len(new_content)} out of {len(content)}, start dump ...")
    json.dump(new_content, open(out_file, "w"), indent=2)
