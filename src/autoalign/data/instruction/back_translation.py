# modifyed from https://github.com/Spico197/Humback
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from functools import partial
from datasets import Dataset
from utils import dump_json, load_jsonlines

from autoalign.conversation import Role
from autoalign.conversation_instr import InstructionGenerateConversation


def apply_template(conv, conv_template_name, key_name):
    """tokenize conversation and prepare labels for loss calculation"""
    # get conversation template
    conversation = InstructionGenerateConversation.from_template(conv_template_name)
    conversation.swap_eos_token()
    # fill in messages from single conv data point
    conversation.append_message(role=Role.ASSISTANT, message=conv[key_name])

    # tokenize conversation
    applied_message = conversation.get_conversation_str(add_generation_prompt="human")
    return {key_name: applied_message}


def main(args):
    llm = LLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    raw_data = load_jsonlines(args.data_filepath)
    dataset = Dataset.from_list(raw_data)

    dataset = dataset.map(
        partial(
            apply_template,
            conv_template_name="vicuna_v1.1",
            key_name=args.prompt_column_name,
        ),
        remove_columns=list(dataset.features),
        num_proc=8,
    )
    prompts = [data[args.prompt_column_name] for data in dataset]
    dump_jsonl = []

    rounds = int(np.ceil(len(prompts) / args.save_cycle))

    for rnd in range(rounds):
        length = min(args.save_cycle, len(prompts) - rnd * args.save_cycle)
        batches = int(np.ceil(length / args.batch_size))
        for batch in range(batches):
            start = rnd * args.save_cycle + batch * args.batch_size
            end = rnd * args.save_cycle + min(
                args.save_cycle, (batch + 1) * args.batch_size
            )

            results = llm.generate(
                prompts[start:end], use_tqdm=True, sampling_params=sampling_params
            )
            for result in results:
                dump_jsonl.append(
                    {
                        "id": len(dump_jsonl),
                        "conversations": [
                            {"from": "human", "value": result.outputs[0].text}
                        ],
                    }
                )
        dump_json(dump_jsonl, args.save_filepath)

    # 07:24 / 100 prompts on one GPU
    # results = []
    # for prompt in tqdm(prompts):
    #     result = llm.generate(prompt, use_tqdm=False, sampling_params=sampling_params)
    #     results.append(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--save_filepath", type=str)
    parser.add_argument("--prompt_column_name", type=str, default="instruction")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--save_cycle", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    main(args)
