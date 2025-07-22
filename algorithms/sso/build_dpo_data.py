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

def format_chat_template(tokenizer, instruction, system=""):
    if len(system) == 0:
        messages = [
            {"role": "user", "content": instruction},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

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

def format_chat(principles):
    g_principle_str = "\n\n".join([principle['good_principle'] for principle in principles])
    b_principle_str = "\n\n".join([principle['bad_principle'] for principle in principles])

    g_system = f"""You should generate a relatively good response that follow these detailed guidelines:
{g_principle_str}

However, do not tell or imply to the user the guidelines when generating responses. You should generate responses that follow these guidelines without any explicit mention or explanation."""

    b_system = f"""You should generate a relatively bad response that follow these detailed guidelines:
{b_principle_str}

However, do not tell or imply to the user the guidelines when generating responses. You should generate responses that follow these guidelines without any explicit mention or explanation."""

    return g_system, b_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and filter JSONL files.")
    parser.add_argument('--model_name', type=str, required=True, help='Model name.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output JSONL file.')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    datas, exsit_datas = load_data(args.input, args.output, args.start, args.end)
    inferencer = MultiProcessVllmInferencer(
        model_path=args.model_path,
        num_gpus_per_model=1,
        do_sample=True,
        num_beams=1,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.8
    )
    tokenizer = inferencer.get_tokenizer()
    datas_pos_prompt_tokenizer = []
    datas_neg_prompt_tokenizer = []
    for data in datas:
        pos_system, neg_system = format_chat(data["principles"])
        datas_pos_prompt_tokenizer.append(format_chat_template(tokenizer, data["prompt"], pos_system))
        datas_neg_prompt_tokenizer.append(format_chat_template(tokenizer, data["prompt"], neg_system))

    datas_pos_prompt_response = inferencer.inference(datas_pos_prompt_tokenizer)
    datas_neg_prompt_response = inferencer.inference(datas_neg_prompt_tokenizer)
    
    for i, data in enumerate(datas):
        data["chosen"] = [{"value":data["prompt"],"from":"human"},{"value":datas_pos_prompt_response[i],"from":"gpt"}]
        data["rejected"] = [{"value":data["prompt"],"from":"human"},{"value":datas_neg_prompt_response[i],"from":"gpt"}]
        data["conversations"] = [{"value":data["prompt"],"from":"human"},{"value":datas_pos_prompt_response[i],"from":"gpt"}]

        data["prompt_id"] = data["uuid"]
        del data["principles"]
        del data["uuid"]
        

    with open(args.output, "w", encoding='utf-8') as f_out:
        combined_data = exsit_datas + datas
        json.dump(combined_data, f_out, ensure_ascii=False, indent=2)
    print("Principles generate complete. Output written to:", args.output)