import os

import datasets
from datasets import DatasetDict


def train_preprocess(item):
    samples = [item["chosen"], item["rejected"]]
    for sample in samples:
        for msg in sample:
            role = msg.pop("role")
            msg["from"] = "gpt" if role == "assistant" else "human"
            msg["value"] = msg.pop("content")

    return {
        "chosen": samples[0],
        "rejected": samples[1],
    }


def prior_preprocess(item):
    prompt = item["prompt"]
    for msg in prompt:
        role = msg.pop("role")
        msg["from"] = "gpt" if role == "assistant" else "human"
        msg["value"] = msg.pop("content")
    chosen, rejected = prompt.copy(), prompt.copy()
    chosen.append({"value": item["chosen"], "from": "gpt"})
    rejected.append({"value": item["rejected"], "from": "gpt"})

    return {
        "chosen": chosen,
        "rejected": rejected,
    }


def bench_preprocess(item):
    prompt = [{"value": item["prompt"], "from": "human"}]
    chosen, rejected = prompt.copy(), prompt.copy()
    chosen.append({"value": item["chosen"], "from": "gpt"})
    rejected.append({"value": item["rejected"], "from": "gpt"})
    return {
        "chosen": chosen,
        "rejected": rejected,
    }


if __name__ == "__main__":
    train_dataset: DatasetDict = datasets.load_dataset(
        "argilla/ultrafeedback-binarized-preferences-cleaned"
    )
    train_dataset = train_dataset["train"].map(train_preprocess)
    # Available at the generation of rm data example "data/ultra_binary.jsonl"
    train_dataset = train_dataset.select(range(8))
    train_dataset.to_json("data/ultra_binary.jsonl")

    os.makedirs("data/eval", exist_ok=True)
    prior_datasets = datasets.load_dataset("allenai/preference-test-sets")
    for name in ["shp", "anthropic_helpful", "summarize"]:
        prior_dataset = prior_datasets[name].map(prior_preprocess)
        # Available at the generation of prior data example in "data/eval/rm"
        prior_dataset = prior_dataset.select(range(8))
        prior_dataset.to_json(f"data/eval/rm/{name}.jsonl")

    bench_dataset = datasets.load_dataset("allenai/reward-bench")
    bench_dataset = bench_dataset["filtered"].map(bench_preprocess)
    # Available at the generation of bench data example "data/eval/rm/rewardbench.jsonl"
    bench_dataset = bench_dataset.select(range(8))
    bench_dataset = bench_dataset.to_json("data/eval/rm/rewardbench.jsonl")
