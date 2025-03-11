import argparse
from autoalign.conversation import Conversation
import json
import os
import random

from autoalign.inference.inferencer import (
    MultiProcessHFInferencer,
    MultiProcessVllmInferencer,
)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
    parser.add_argument("--output-dir", default="./outputs/")
    parser.add_argument("--num_gpus_per_model", type=int, default=1)
    parser.add_argument("--system-message", type=str, default=None)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--debug-mode", action="store_true")
    args = parser.parse_args()

    return args


def inference():

    args = parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        all_test_points = json.loads(f.read())
    print(f"Data Length: {len(all_test_points)}")
    if args.debug_mode:
        all_test_points = all_test_points[:1000]

    if args.backend == "hf":
        inferencer = MultiProcessHFInferencer(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens_per_utterance,
            num_beams=1,
            top_p=1,
            temperature=0,
            do_sample=False,
        )

    elif args.backend == "vllm":
        inferencer = MultiProcessVllmInferencer(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens_per_utterance,
            num_gpus_per_model=args.num_gpus_per_model,
            num_beams=1,
            top_p=1,
            temperature=0,
        )

    test_file_name = args.test_file.split("/")[-1]
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    output_file_name = f"{args.source}_{test_file_name}"
    output_file_path = os.path.join(args.output_dir, args.model_name, output_file_name)
    print(output_file_path)

    all_convs = []

    for d in all_test_points:
        conv = Conversation.from_template(args.template)
        if d["conversations"][-1]["from"] == "gpt":
            d["conversations"] = d["conversations"][:-1]
        # override the all system message if provided
        if args.system_message:
            conv.system_message = args.system_message
            # clean the last message if it is from gpt
            conv.fill_in_messages(d, replace_conv_system_message=False)
        else:
            conv.fill_in_messages(d)
        all_convs.append(conv)

    turn_inputs = [
        conv.get_conversation_str(add_generation_prompt=True) for conv in all_convs
    ]

    idx = random.choice(range(len(all_test_points)))

    print("===============")
    print(f"Rendered Sample[{idx}]: {turn_inputs[idx]}")
    print("===============")

    all_responses = inferencer.inference(turn_inputs)

    assert len(all_test_points) == len(all_responses)

    all_responses = [r.lstrip() for r in all_responses]
    print(f"Sample Response[{idx}]: {all_responses[idx]}")

    for conv, res in zip(all_convs, all_responses):
        res = res.rstrip(inferencer.get_tokenizer().eos_token)
        conv.append_message("gpt", res)

    with open(output_file_path, "w", encoding="utf-8") as f:
        all_outputs = []
        for idx, (conv, d) in enumerate(zip(all_convs, all_test_points)):
            messages = conv.get_messages()
            d["conversations"].append({"from": "gpt", "value": messages[-1][1]})
            all_outputs.append(
                {
                    "id": d["id"] if "id" in d else f"{test_file_name}_{idx}",
                    "conversations": d["conversations"],
                    "source": args.source,
                }
            )

        f.write(json.dumps(all_outputs, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    inference()
