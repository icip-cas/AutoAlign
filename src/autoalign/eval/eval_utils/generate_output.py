import datasets
import json
import time
import tqdm
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def build_data(datas, batch_size):
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        skip_special_tokens=True,
        max_tokens=1024)
    sorted_datas = sorted(datas, key=lambda x: len(x['instruction']))
    dealdatas = []
    batch_num = (len(sorted_datas)-1)// batch_size + 1
    for i in tqdm.tqdm(range(batch_num)):
        batch_datas = sorted_datas[i*batch_size:(i+1)*batch_size]
        prompts = []
        for data in batch_datas:
            messages = [{"role": "user", "content": data['instruction']}]
            prompts.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            ))
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for j, data in enumerate(batch_datas):
            data['prompt'] = data['instruction']
            data["output"] = outputs[j].outputs[0].text.strip()
            dealdatas.append(data.copy())
    return dealdatas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--name', type=str)
    args = parser.parse_args()
    model = args.model
    batch_size = args.batch_size
    output_path = args.output_path
    name = args.name

    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    llm = LLM(model=model,
            enforce_eager=True,
            tensor_parallel_size=torch.cuda.device_count()
            )
    tokenizer = AutoTokenizer.from_pretrained(model)
    datas = build_data(eval_set, batch_size=batch_size)
    for data in datas:
        data["generator"] = name
    with open(output_path, 'w') as f:
        json.dump(datas, f, indent=4)
