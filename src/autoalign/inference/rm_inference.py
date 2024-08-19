import argparse
import json
import os
import random
from datetime import timedelta

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from autoalign.conversation import Conversation


class MultiProcessHFInferencer:
    def __init__(
        self,
        model_path: str,
        nccl_timeout: int = 64000,
    ):
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=nccl_timeout))
        self.accelerator = Accelerator(kwargs_handlers=[kwargs])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # get model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map={"": self.accelerator.process_index},
        )

    def inference(self, data):
        # for each process
        with self.accelerator.split_between_processes(data) as data_on_process:
            rewards = []
            with torch.no_grad():
                for d in tqdm(data_on_process):
                    # currently only support bs=1
                    inputs = self.tokenizer(d, return_tensors="pt").to(self.device)
                    outputs = self.model(**inputs)
                    rewards.extend(outputs.logits.mean(dim=-1).tolist())

        # gather responses
        gathered_rewards = gather_object(rewards)

        return gathered_rewards

    def get_tokenizer(self):
        return self.tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output-dir", default="./outputs/")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--system-message", type=str, default=None)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--debug-mode", action="store_true")
    args = parser.parse_args()
    return args


def inference():
    args = parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        all_test_points = json.loads(f.read())

    if args.debug_mode:
        all_test_points = all_test_points[:1000]

    inferencer = MultiProcessHFInferencer(model_path=args.model_path)

    test_file_name = args.test_file.split("/")[-1]
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    output_file_name = f"{args.source}_{test_file_name}"
    output_file_path = os.path.join(args.output_dir, args.model_name, output_file_name)
    print(output_file_path)

    all_rewards = []

    for d in all_test_points:
        reward = Conversation.from_template(args.template)
        reward.system_message = args.system_message if args.system_message else None
        reward.fill_in_messages(d)
        all_rewards.append(reward)

    turn_inputs = [
        conv.get_conversation_str(add_generation_prompt=True) for conv in all_rewards
    ]

    idx = random.choice(range(len(all_test_points)))

    print("===============")
    print(f"Rendered Sample[{idx}]: {turn_inputs[idx]}")
    print("===============")

    all_rewards = inferencer.inference(turn_inputs)

    assert len(all_test_points) == len(all_rewards)

    print(
        f"Sample Reward[{idx}]: {turn_inputs[idx]}\n{'|' + '-' * 20 + '|'}\n\n{all_rewards[idx]}"
    )

    with open(output_file_path, "w", encoding="utf-8") as f:
        all_outputs = []
        for idx, (reward, d) in enumerate(zip(all_rewards, all_test_points)):
            all_outputs.append(
                {
                    "id": d["id"] if "id" in d else f"{test_file_name}_{idx}",
                    "conversations": d["conversations"],
                    "source": args.source,
                    "reward": reward,
                }
            )

        f.write(json.dumps(all_outputs, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    inference()
