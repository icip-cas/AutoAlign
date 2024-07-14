import argparse
from autoalign.conversation import Conversation
import json
from tqdm import tqdm
import os

from autoalign.inference.inferencer import MultiProcessHFInferencer

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
    parser.add_argument("--output-dir", default="./outputs/")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--system-message", type=str, default="You are a helpful assistant.")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--debug-mode", action="store_true")
    args = parser.parse_args()

    return args

def inference():

    args = parse_args()

    def preprocess(d):
        d["instruction"] = d["conversations"][0]["value"]
        return d

    with open(args.test_file, "r", encoding="utf-8") as f:
        all_test_points = json.loads(f.read())

    all_test_points = [preprocess(d) for d in all_test_points]

    if args.debug_mode:
        all_test_points = all_test_points[:100]

    inferencer = MultiProcessHFPipelineInferencerAccelerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens_per_utterance,
        num_beams=1,
        top_p=1,
        temperature=0
    )

    # inferencer = MultiProcessVllmInferencer(
    #     model_path=args.model_path,
    #     max_new_tokens=args.max_new_tokens_per_utterance,
    #     num_beams=1,
    #     top_p=1,
    #     temperature=0
    # )

    test_file_name = args.test_file.split("/")[-1]
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    output_file_name = f"{args.source}_{test_file_name}"
    output_file_path = os.path.join(args.output_dir, args.model_name, output_file_name)
    print(output_file_path)

    unrelated_idx = 0
    conv_idx = 0
    all_convs = []
        
    for d in all_test_points:
        conv = get_conv_template(args.template)
        conv.system_message = args.system_message
        if "system" in d and d["system"]:
            conv.system_message = d["system"]
        conv.append_message(conv.roles[0], d["instruction"])
        conv.append_message(conv.roles[1], None)
        all_convs.append(conv)

    turn_inputs = [conv.get_prompt() for conv in all_convs]

    print("===============")
    print(turn_inputs[0])
    print("===============")
        
    all_responses = inferencer.inference(turn_inputs)

    all_responses = [r.lstrip() for r in all_responses]
    for conv, res in zip(all_convs, all_responses):
        res = res.rstrip(inferencer.get_tokenizer().eos_token)
        conv.append_message(conv.roles[1], res)

    with open(output_file_path, "w", encoding="utf-8") as f:
        all_outputs = []
        for idx, (conv, d) in enumerate(zip(all_convs, all_test_points)):
            messages = [msg for msg in conv.messages if msg[1] is not None]
            all_outputs.append(
                {
                    "id": d["id"] if "id" in d else f"{test_file_name}_{idx}",
                    "system": conv.system_message, 
                    "conversations": [
                        {
                            "from": "human",
                            "value": messages[-2][1]
                        },
                        {
                            "from": "gpt",
                            "value": messages[-1][1]
                        }
                    ],
                    "source": args.source 
                }
            )
        
        f.write(json.dumps(all_outputs, ensure_ascii=False))

if __name__ == "__main__":
    inference()