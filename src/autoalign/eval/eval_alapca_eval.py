import datasets
import os
from inferencer import HFInferencer
from argparse import ArgumentParser
from fastchat.model import get_conversation_template
from modeling import LlamaResidualPlus

from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument("--residual-model", action="store_true")
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--template-name", type=str, required=True)
parser.add_argument("--model-path", type=str)
parser.add_argument("--pretrained-model-path", type=str)
parser.add_argument("--max-new-tokens", type=int, default=1024)
parser.add_argument("--system-message", type=str, default=None)
parser.add_argument("--output-dir", type=str, default="outputs/alpaca-eval")
args = parser.parse_args()

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]

eval_set = eval_set

model = LlamaResidualPlus.from_pretrained(
    model_name_or_path=args.model_path,
    pretrained_model_name_or_path=args.pretrained_model_path,
)

model.to_cuda()

tokenizer = AutoTokenizer.from_pretrained(
    args.model_path,
)

inferencer = HFInferencer(
    model=model,
    tokenizer=tokenizer
)

sampling_params = {
    "temperature": 0.8,
    "do_sample": True,
    "max_new_tokens": args.max_new_tokens,
}

for example in eval_set:
    conv = get_conversation_template(args.template_name)
    if args.system_message is not None:
        conv.system_message = args.system_message
    conv.append_message(conv.roles[0], example["instruction"])
    conv.append_message(conv.roles[1], None)
    inputs = conv.get_prompt()
    example["output"] = inferencer.inference(
        inputs,
        **sampling_params
    )
    example["generator"] = args.model_name

os.makedirs(args.output_dir, exist_ok=True)

output_file_path = os.path.join(args.output_dir, f"{args.model_name}.json")

eval_set.to_json(output_file_path)
