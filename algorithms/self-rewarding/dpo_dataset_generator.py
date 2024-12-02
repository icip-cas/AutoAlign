import json
import torch
import random
import re
import multiprocessing
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from copy import deepcopy
import argparse
import os

from templates import (
    DEFAULT_TASK_GENERATION_PROMPT,
    DEFAULT_TASK_PROMPT,
    DEFAULT_REWARD_REGEX_TEMPLATE,
)
from autoalign.conversation import Conversation
from autoalign.prompts.judge import META_JUDGE_EN


def worker(
    model_name: str,
    questions: list[dict],
    num_choices: int,
    gpu_ids: list[int],
    input_hook_fn,
    output_hook_fn,
    **kwargs,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(",".join(gpu_ids))
    from autoalign.inference.inferencer import (
        HFInferencer,
    )

    inferencer = HFInferencer(model_name_or_path=model_name)
    print("Inferencer loaded.")
    infer_kwargs = deepcopy(kwargs)
    if "template_name" in infer_kwargs:
        infer_kwargs.pop("template_name")
    if "reward_regex_str" in infer_kwargs:
        infer_kwargs.pop("reward_regex_str")
    answers = []
    for question in tqdm(questions):
        assert "index" in question
        choices = []
        # TODO: In different stage, pass differnent hook to parse the data and get conversation str.
        prompt = input_hook_fn(question, **kwargs)
        for i in range(num_choices):
            torch.manual_seed(i)
            try:
                output = inferencer.inference(prompt, **infer_kwargs)
                output = output_hook_fn(output, **kwargs)
                if output is None:
                    output = float(random.randint(0, 5))
                if output > 5.0 or output < 0.0:
                    output = float(random.randint(0, 5))
            except RuntimeError as e:
                print(e)
                print("ERROR question ID: ", question["index"])
                output = "ERROR"
            choices.append({"index": i, "output": output})
        answers.append({"index": question["index"], "choices": choices})
    return answers


def parallel_inference(
    model_name: str,
    questions: list[dict],
    num_choices: int,
    input_hook_fn,
    output_hook_fn,
    num_gpus_per_model: int,
    **kwargs,
):
    # random shuffle the questions to balance the loading
    random.shuffle(questions)
    num_gpus_total = 8
    print(num_gpus_total)
    process_num = num_gpus_total // num_gpus_per_model
    gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    # os.environ.pop('CUDA_VISIBLE_DEVICES')
    gpus_splits = [
        gpus[i : i + num_gpus_per_model]
        for i in range(0, len(gpus), num_gpus_per_model)
    ]
    print(gpus_splits)
    if process_num == 1:
        results = worker(
            model_name,
            questions,
            num_choices,
            gpus_splits[0],
            input_hook_fn,
            output_hook_fn,
            **kwargs,
        )
        return sorted(results, key=lambda x: x["index"])
    elif process_num > 1:
        chunk_size = len(questions) // process_num
        if len(questions) % process_num == 0:
            with multiprocessing.Pool(processes=process_num) as pool:
                results = [
                    pool.apply_async(
                        worker,
                        args=(
                            model_name,
                            questions[
                                i
                                * chunk_size : min(
                                    i * chunk_size + chunk_size, len(questions)
                                )
                            ],
                            num_choices,
                            gpus_splits[i],
                            input_hook_fn,
                            output_hook_fn,
                        ),
                        kwds=kwargs,
                    )
                    for i in range(process_num)
                ]
                for result in results:
                    result.wait()
                gathered_responses = [output.get() for output in results]
                print(gathered_responses)
                gathered_responses = [
                    item for sublist in gathered_responses for item in sublist
                ]
                gathered_responses = sorted(
                    gathered_responses, key=lambda x: x["index"]
                )
                return gathered_responses
        else:
            flags = [i for i in range(0, len(questions), chunk_size)]
            for i in range(len(flags)):
                if i < len(questions) % process_num:
                    flags[i] += i
                else:
                    flags[i] += len(questions) % process_num
            assert process_num == len(flags) - 1
            with multiprocessing.Pool(processes=process_num) as pool:
                results = [
                    pool.apply_async(
                        worker,
                        args=(
                            model_name,
                            questions[flags[i] : flags[i + 1]],
                            num_choices,
                            gpus_splits[i],
                            input_hook_fn,
                            output_hook_fn,
                        ),
                        kwds=kwargs,
                    )
                    for i in range(len(flags) - 1)
                ]
                for result in results:
                    result.wait()
                gathered_responses = [output.get() for output in results]
                gathered_responses = [
                    item for sublist in gathered_responses for item in sublist
                ]
                gathered_responses = sorted(
                    gathered_responses, key=lambda x: x["index"]
                )
                return gathered_responses


def format_examples(message, **kwargs):
    assert "examples" in message
    task_generation_prompt = DEFAULT_TASK_GENERATION_PROMPT
    for example in message["examples"]:
        task_generation_prompt += DEFAULT_TASK_PROMPT.format(item=example)
    return task_generation_prompt


def extract_prompts(answer, **kwargs):
    prompts = []
    while True:
        pattern = "<task>"
        start = answer.find(pattern)
        if start == -1:
            break
        end = answer.find("</task>")
        if end == -1:
            break
        prompts.append(answer[start + len(pattern) : end])
        answer = answer[end + len("</task>") :]
    print("Prompts extracted:")
    print(prompts)
    return prompts


def format_conversation(conv, **kwargs):
    template_name = kwargs.get("template_name")
    conversation = Conversation.from_template(template_name)
    conversation.fill_in_messages(conv)
    return conversation.get_conversation_str(add_generation_prompt=True)


def format_prompt(pair, **kwargs):
    prompt_reward_model = META_JUDGE_EN.format(
        instruction=pair["instruct"], response=pair["response"]
    )
    return prompt_reward_model


import jinja2

jinja2_env = jinja2.Environment()


def find_variables_from_jinja_template(template: str):
    from jinja2 import meta

    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {
        "reward"
    }, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(
        reward="([-+]?[0-9]+\.?[0-9]*)"
    )
    # @always(lambda: randrange(0, 10))
    return reward_regex_str


def parse_reward_fn(llm_response: str, **kwargs) -> float:
    reward_regex_str = kwargs.get("reward_regex_str")
    print(reward_regex_str)
    result = re.search(rf"{reward_regex_str}", llm_response)
    if result is None:
        return None
    try:
        result.group(1)
    except Exception as e:
        print("Encounted {} while parsing reward function.".format(e))
        return None
    return float(result.group(1))


def identity(x, **kwargs):
    return x


class DPODatasetGenerator:
    def __init__(self, model_name, ift_dataset, dpo_dataset):
        self.model_name = model_name
        self.ift_dataset = Dataset.from_list(json.load(open(ift_dataset, "r")))
        self.dpo_dataset_file = open(dpo_dataset, "w", encoding="utf-8")

    def generate_prompts(self, num_prompts: int, num_examples: int = 8):
        task_generation_prompts = []

        for index in range(num_prompts):
            rand_examples_idxs = random.sample(
                range(0, len(self.ift_dataset)), k=num_examples
            )
            examples = []
            for idx in rand_examples_idxs:
                examples.append(self.ift_dataset[idx]["conversations"][0]["value"])
            task_generation_prompts.append({"index": index, "examples": examples})
        raw_task_prompts = parallel_inference(
            model_name=self.model_name,
            questions=task_generation_prompts,
            num_choices=1,
            input_hook_fn=format_examples,
            output_hook_fn=extract_prompts,
            num_gpus_per_model=args.num_gpus_per_model,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=512,
        )
        task_prompts = []
        for raw_prompt in raw_task_prompts:
            task_prompts.extend(raw_prompt["choices"][0]["output"])
        rem_dup_task_prompts = list(set(task_prompts))
        rem_dup_task_prompts = [
            {"index": idx, "conversations": [{"from": "human", "value": prompt}]}
            for idx, prompt in enumerate(rem_dup_task_prompts)
        ]
        json.dump(
            rem_dup_task_prompts,
            open("data/test/task_prompts.json", "w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
        # form:
        return rem_dup_task_prompts

    def generate_responses(self, prompts, template_name, repeat_time=4):

        # TODO: 需要修一下Inferencer: 用ray并行，并且保证输出的response顺序是输入的prompt的顺序
        prompts = sorted(prompts, key=lambda x: x["index"])
        responses = parallel_inference(
            model_name=self.model_name,
            questions=prompts[0:8],
            num_choices=repeat_time,
            input_hook_fn=format_conversation,
            output_hook_fn=identity,
            num_gpus_per_model=args.num_gpus_per_model,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
            template_name=template_name,
        )
        prompt_response_pairs = []
        prompt_index = [prompt["index"] for prompt in prompts]
        response_index = [response["index"] for response in responses]
        print(len(prompt_index))
        print(prompt_index)
        print(len(response_index))
        print(response_index)
        for prompt, response in zip(prompts, responses):
            print(prompt["index"])
            print(response["index"])
            assert prompt["index"] == response["index"]
            for choice in response["choices"]:
                prompt_response_pairs.append(
                    {
                        "index": prompt["index"] * repeat_time + choice["index"],
                        "instruct": prompt["conversations"][0]["value"],
                        "response": choice["output"],
                    }
                )
        json.dump(
            prompt_response_pairs,
            open("data/test/prompts_responses.json", "w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
        return prompt_response_pairs

    def generate_scores(self, instr_resp_pairs, repeat_time=4, score_time=3):

        reward_regex = create_parse_reward_fn(DEFAULT_REWARD_REGEX_TEMPLATE)
        raw_scores = parallel_inference(
            model_name=self.model_name,
            questions=instr_resp_pairs,
            num_choices=score_time,
            input_hook_fn=format_prompt,
            output_hook_fn=parse_reward_fn,
            num_gpus_per_model=args.num_gpus_per_model,
            temperature=0.7,
            top_p=0.9,
            max_length=8192,
            max_new_tokens=512,
            reward_regex_str=reward_regex,
        )

        instr_resp_pairs = sorted(instr_resp_pairs, key=lambda x: x["index"])
        conv_score = []
        # TODO: 每个response打分3遍
        for pair, score in zip(instr_resp_pairs, raw_scores):
            assert pair["index"] == score["index"]
            assert len(score["choices"]) == score_time
            # valid_score = []
            # for score in
            score = sum([choice["output"] for choice in score["choices"]]) / score_time
            new_pair = deepcopy(pair)
            new_pair["score"] = score
            conv_score.append(new_pair)

        json.dump(
            conv_score,
            open("data/oasst1/en_oasst_first_turn_scores.json", "w", encoding="utf-8"),
            indent=2,
            ensure_ascii=False,
        )
        return conv_score

    def generate_preferences(self, conv_scores, repeat_time=4):
        formed_data = []
        for i in range(len(conv_scores)):
            if i % repeat_time == 0:
                formed_data.append(
                    {
                        "index": i // 4,
                        "instruct": conv_scores[i]["instruct"],
                        "response": [conv_scores[i]["response"]],
                        "score": [conv_scores[i]["score"]],
                    }
                )
            else:
                formed_data[i // 4]["response"].append(conv_scores[i]["response"])
                formed_data[i // 4]["score"].append(conv_scores[i]["score"])
        preference_data = []
        for data in formed_data:
            sorted_index = np.argsort(np.array(data["score"]))
            if data["score"][sorted_index[0]] < data["score"][sorted_index[-1]]:
                preference_data.append(
                    {
                        "prompt": data["instruct"],
                        "prompt_id": len(preference_data),
                        "chosen": [
                            {"value": data["instruct"], "from": "human"},
                            {
                                "value": data["response"][sorted_index[-1]],
                                "from": "gpt",
                            },
                        ],
                        "rejected": [
                            {"value": data["instruct"], "from": "human"},
                            {"value": data["response"][sorted_index[0]], "from": "gpt"},
                        ],
                        "conversations": [
                            {"value": data["instruct"], "from": "human"},
                            {
                                "value": data["response"][sorted_index[-1]],
                                "from": "gpt",
                            },
                        ],
                        "score_chosen": data["score"][sorted_index[-1]],
                        "score_rejected": data["score"][sorted_index[0]],
                    }
                )
        json.dump(
            preference_data,
            open("./test/preference_data.json", "w", encoding="utf-8"),
            ensure_ascii=False,
        )
        return preference_data


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
        help="The input question file path.",
    )

    args = parser.parse_args()
    print("generator_initing")
    generator = DPODatasetGenerator(
        args.model_path,
        ift_dataset="data/test/seed.json",
        dpo_dataset="data/test/dpo_dataset.json",
    )
    print("generator_inited")
    # generator.generate_prompts(num_prompts=8)

    # prompts = json.load(open("data/test/task_prompts.json", 'r'))
    # generator.generate_responses(prompts=prompts, template_name="llama-3-instruct")

    pairs = json.load(open("data/oasst1/en_oasst_first_turn_formed.json", "r"))
    generator.generate_scores(instr_resp_pairs=pairs)

    # pairs = json.load(open("data/oasst1/en_oasst_first_turn_formed.json", 'r'))
    # generator.generate_scores(instr_resp_pairs=pairs)
