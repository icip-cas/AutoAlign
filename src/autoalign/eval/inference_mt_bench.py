import argparse
import json
import os
import random
import time

import torch
from tqdm import tqdm

from .utils import load_questions, temperature_config
from autoalign.conversation import Conversation, Role
from autoalign.inference.inferencer import HFInferencer


def _run_mt_bench_eval(
    model_path,
    model_id,
    template_name,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    inferencer = HFInferencer(
        model_path
    )

    get_answers_func = get_model_answers

    ans_handles = []
    for i in tqdm(range(0, len(questions))):
        ans_handles.append(
            get_answers_func(
                inferencer,
                model_id,
                template_name,
                questions[i],
                answer_file,
                max_new_token,
                num_choices,
            )
        )


@torch.inference_mode()
def get_model_answers(
    inferencer,
    model_id,
    template_name,
    question,
    answer_file,
    max_new_token,
    num_choices,
):

    tokenizer = inferencer.get_tokenizer()

    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    # print("temperature: ", temperature)

    choices = []
    for i in range(num_choices):
        torch.manual_seed(i)
        conv = Conversation.from_template(template_name)
        turns = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(Role.HUMAN, qs)
            prompt = conv.get_conversation_str(add_generation_prompt=True)

            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            # print("Input:", prompt)

            # some models may error out when generating long outputs
            try:
                if not do_sample:
                    output = inferencer.inference(
                        prompt, 
                        do_sample=do_sample, 
                        max_new_tokens=max_new_token
                    )
                else:
                    output = inferencer.inference(
                        prompt, 
                        temperature=temperature, 
                        do_sample=do_sample, 
                        max_new_tokens=max_new_token
                    )

                if conv.template.stop_str and isinstance(conv.template.stop_str, list):
                    stop_str_indices = sorted(
                        [output.find(stop_str) for stop_str in conv.template.stop_str if output.find(stop_str) > 0]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.template.stop_str and output.find(conv.template.stop_str) > 0:
                    output = output[: output.find(conv.template.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

            except RuntimeError as e:
                print(e)
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            print("Output:", output)

            print("======================")

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "a") as fout:
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": None,
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for ll in fin:
            qid = json.loads(ll)["question_id"]
            answers[qid] = ll

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--model-id", type=str, required=True, help="A custom name for the model.")
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument("--question-end", type=int, help="A debug option. The end index of questions.")
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )

    args = parser.parse_args()

    question_file = "data/eval/mt-bench/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/eval/mt-bench/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    _run_mt_bench_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
    )

    reorg_answer_file(answer_file)
