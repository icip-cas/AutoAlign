import argparse
from autoalign.conversation import Conversation
import json
import os

from autoalign.inference.inferencer import MultiProcessHFInferencer, MultiProcessVllmInferencer

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--max-new-tokens-per-utterance", type=int, default=1024)
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
        all_test_points = all_test_points[:2]

    if args.backend == "hf":
        inferencer = MultiProcessHFInferencer(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens_per_utterance,
            num_beams=1,
            top_p=1,
            temperature=0,
            do_sample=False
        )

    elif args.backend == "vllm":
        inferencer = MultiProcessVllmInferencer(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens_per_utterance,
            num_beams=1,
            top_p=1,
            temperature=0
        )

    test_file_name = args.test_file.split("/")[-1]
    os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
    output_file_name = f"{args.source}_{test_file_name}"
    output_file_path = os.path.join(args.output_dir, args.model_name, output_file_name)
    print(output_file_path)

    all_convs = []
        
    for d in all_test_points:
        conv = Conversation.from_template(
            args.template,
            overwrite_system_message=args.system_message if args.system_message else None
        )
        # clean the last message if it is from gpt
        if d["conversations"][-1]["from"] == "gpt":
            d["conversations"] = d["conversations"][:-1]
        conv.fill_in_messages(d)
        all_convs.append(conv)

    turn_inputs = [conv.get_conversation_str() for conv in all_convs]

    print("===============")
    print(turn_inputs[0])
    print("===============")
        
    all_responses = inferencer.inference(turn_inputs)

    all_responses = [r.lstrip() for r in all_responses]
    for conv, res in zip(all_convs, all_responses):
        res = res.rstrip(inferencer.get_tokenizer().eos_token)
        conv.append_message("gpt", res)

    with open(output_file_path, "w", encoding="utf-8") as f:
        all_outputs = []
        for idx, (conv, d) in enumerate(zip(all_convs, all_test_points)):
            messages = conv.get_messages()
            all_outputs.append(
                {
                    "id": d["id"] if "id" in d else f"{test_file_name}_{idx}",
                    "conversations": d["conversations"].append(
                        {
                            "from": "gpt",
                            "value": messages[-1][1]
                        }
                    ),
                    "source": args.source 
                }
            )
        
        f.write(json.dumps(all_outputs, ensure_ascii=False))

if __name__ == "__main__":
    inference()