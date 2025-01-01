import argparse
import json
import os
import random
import time

import torch
from tqdm import tqdm

from autoalign.eval.utils import load_questions, temperature_config
from autoalign.conversation import Conversation, Role
from autoalign.inference.inferencer import HFInferencer
from vllm import LLM, SamplingParams


def inference_mt_bench(
    model_path,
    model_id,
    template_name,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    backend,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)
    num_gpus_total = torch.cuda.device_count()
    process_num = num_gpus_total // num_gpus_per_model
    ans_handles = []
    if process_num == 1:
        get_answers_func = get_model_answers
        get_answers_func(
            model_path,
            model_id,
            template_name,
            questions,
            answer_file,
            max_new_token,
            num_choices,
            backend,
        )
    elif process_num > 1:
        import ray

        ray.init()
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
        chunk_size = len(questions) // process_num
        if len(questions) % process_num == 0:
            for i in tqdm(range(0, len(questions), chunk_size)):
                ans_handles.append(
                    get_answers_func(
                        model_path,
                        model_id,
                        template_name,
                        questions[i : min(i + chunk_size, len(questions))],
                        answer_file,
                        max_new_token,
                        num_choices,
                        backend,
                    )
                )
        else:
            flags = [i for i in range(0, len(questions), chunk_size)]
            for i in range(len(flags)):
                if i < len(questions) % process_num:
                    flags[i] += i
                else:
                    flags[i] += len(questions) % process_num
            for i in range(len(flags) - 1):
                ans_handles.append(
                    get_answers_func(
                        model_path,
                        model_id,
                        template_name,
                        questions[flags[i] : flags[i + 1]],
                        answer_file,
                        max_new_token,
                        num_choices,
                        backend,
                    )
                )

        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    template_name,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    backend="vllm",
):
    if backend == "hf":
        inferencer = HFInferencer(model_path)
        tokenizer = inferencer.get_tokenizer()
    elif backend == "vllm":
        llm = LLM(model=model_path, gpu_memory_utilization=0.95)
        tokenizer = llm.get_tokenizer()
    for question in tqdm(questions):
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
                if backend == "hf":
                    if temperature < 1e-4:
                        do_sample = False
                    else:
                        do_sample = True
                elif backend == "vllm":
                    sampling_params = SamplingParams(
                        temperature=temperature if temperature > 1e-4 else 0.0,
                        max_tokens=max_new_token,
                    )

                # print("Input:", prompt)

                # some models may error out when generating long outputs
                try:
                    if backend == "hf":
                        output = inferencer.inference(
                            prompt,
                            temperature=temperature if do_sample else 0.0,
                            do_sample=do_sample,
                            max_new_tokens=max_new_token,
                        )
                    elif backend == "vllm":
                        outputs = llm.generate([prompt], sampling_params)
                        output = outputs[0].outputs[0].text

                    if conv.template.stop_str and isinstance(
                        conv.template.stop_str, list
                    ):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.template.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif (
                        conv.template.stop_str
                        and output.find(conv.template.stop_str) > 0
                    ):
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

                # print("Output:", output)

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
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
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
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="The input question file path.",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        help="The chat template used in inference.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        help="The backend used in inference.",
        default="vllm",
    )
    args = parser.parse_args()

    if args.question_file:
        question_file = args.question_file
    else:
        question_file = "data/eval/mt-bench/question.jsonl"

    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/eval/mt-bench/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    inference_mt_bench(
        model_path=args.model_path,
        model_id=args.model_id,
        template_name=args.template_name,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        backend=args.backend,
    )

    reorg_answer_file(answer_file)
