import json
import argparse
import random
import os
from tqdm import tqdm
from prompts import principle_prompt, helpful, concision
from autoalign.inference.inferencer import (
    MultiProcessHFInferencer,
    MultiProcessVllmInferencer,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and filter JSONL files.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    return parser.parse_args()
def load_data(path, out_path="", start=0, end=None):
    exsit_datas = []
    if len(out_path) > 0 and os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            try:
                exsit_datas = json.load(f)
            except Exception as e:
                exsit_datas = [json.loads(line) for line in f if line.strip()]
    exsit_ids = [d['uuid'] for d in exsit_datas]
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            data = [json.loads(line) for line in f if line.strip()]
    
    if 'uuid' not in data[0]:
        for k, d in enumerate(data):
            d["uuid"] = k

    data = data[start: end] if end else data[start:]
    data = [d for d in data if 'uuid' in d and d['uuid'] not in exsit_ids]
    data = data
    return data, exsit_datas

if __name__ == "__main__":
    args = parse_arguments()
    datas, exsit_datas = load_data(args.input, args.output, args.start, args.end)
    inferencer = MultiProcessVllmInferencer(
        model_path=args.model_path,
        num_gpus_per_model=1,
        do_sample=True,
        num_beams=1,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.8,
        stop = ["###"]
    )
    tokenizer = inferencer.get_tokenizer()
    querys = [data["prompt"] for data in datas]
    prompts = [principle_prompt.format(query=query) for query in querys]
    prompt_tokenizer =[tokenizer.apply_chat_template(
                [{"role":"user","content":prompt}],
                tokenize=False,
                add_generation_prompt=True) for prompt in prompts]
    bad_principles = inferencer.inference(prompt_tokenizer)
    good_principle_prompts = [prompt + bad_principles[i] + "\n### Good Guideline\n" for i, prompt in enumerate(prompts)]
    prompt_tokenizer =[tokenizer.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=False,
            add_generation_prompt=True) for prompt in good_principle_prompts]
    good_principles = inferencer.inference(prompt_tokenizer)
    for i, data in  enumerate(datas):
        principle = []
        ### 这里没有考虑random.random() <= 0.25
        if len(bad_principles[i]) > 0 and len(good_principles[i]) > 0:
            principle = [{
                "good_principle": good_principles[i],
                "bad_principle": bad_principles[i]
            }]
        else:
            principle = [helpful]
        data["principles"] = principle

    with open(args.output, "w", encoding='utf-8') as f_out:
        for d in exsit_datas:
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")

        for d in datas:
            f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("Principles generate complete. Output written to:", args.output)
