import torch
import random
import re
import multiprocessing
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import argparse
import os
from utils import (
    identity,
    load_json,
    dump_json,
    get_available_gpu_list,
    post_process_instructs,
)
from templates import (
    DEFAULT_TASK_GENERATION_PROMPT,
    DEFAULT_REWARD_REGEX_TEMPLATE,
    create_parse_reward_fn,
)
from autoalign.conversation import Conversation
from autoalign.prompts.judge import META_JUDGE_EN


def inference(
    model_name: str,
    questions: list[dict],
    num_choices: int,
    gpu_ids: list[int],
    input_hook_fn,
    output_hook_fn,
    **kwargs,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(",".join(gpu_ids))
    answers = []
    if kwargs["backend"] == "hf":
        from autoalign.inference.inferencer import HFInferencer

        inferencer = HFInferencer(model_name_or_path=model_name)
        infer_kwargs = deepcopy(kwargs)
        if "template_name" in infer_kwargs:
            infer_kwargs.pop("template_name")
        if "reward_regex_str" in infer_kwargs:
            infer_kwargs.pop("reward_regex_str")
        if "instruction_pool" in infer_kwargs:
            infer_kwargs.pop("instruction_pool")
        if "ift_dataset" in infer_kwargs:
            infer_kwargs.pop("ift_dataset")
        if "backend" in infer_kwargs:
            infer_kwargs.pop("backend")
        for question in tqdm(questions):
            assert "id" in question
            choices = []
            prompt = input_hook_fn(question, **kwargs)
            for i in range(num_choices):
                torch.manual_seed(i)
                try:
                    output = inferencer.inference(prompt, **infer_kwargs)
                    output = output_hook_fn(output, **kwargs)
                except RuntimeError as e:
                    print(e)
                    print("ERROR question ID: ", question["id"])
                    output = "ERROR"
                choices.append({"id": i, "output": output})
            answers.append({"id": question["id"], "choices": choices})
    elif kwargs["backend"] == "vllm":
        from vllm import LLM
        from vllm.sampling_params import SamplingParams

        os.environ["VLLM_PORT"] = str(
            random.choice(list(range(40000, 50030))) + int(gpu_ids[0])
        )
        llm = LLM(
            model=model_name,
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.95,
        )
        sampling_params = SamplingParams(
            n=num_choices,
            max_tokens=kwargs["max_new_tokens"],
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
        )

        for question in tqdm(questions):
            assert "id" in question
            choices = []
            prompt = input_hook_fn(question, **kwargs)
            outputs = llm.generate(prompt, sampling_params)
            for output in outputs:
                prompt = output.prompt
                for idx, generated_output in enumerate(output.outputs):
                    generated_text = generated_output.text
                    completion = output_hook_fn(generated_text, **kwargs)
                    choices.append({"id": idx, "output": completion})
            answers.append({"id": question["id"], "choices": choices})
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
    try:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    except Exception:
        # print(e)
        gpus = get_available_gpu_list()
    num_gpus_total = len(gpus)
    process_num = min(len(questions), num_gpus_total // num_gpus_per_model)
    # os.environ.pop('CUDA_VISIBLE_DEVICES')
    gpus_splits = [
        gpus[i : i + num_gpus_per_model]
        for i in range(0, len(gpus), num_gpus_per_model)
    ]
    if process_num == 1:
        results = inference(
            model_name,
            questions,
            num_choices,
            gpus_splits[0],
            input_hook_fn,
            output_hook_fn,
            **kwargs,
        )
        return sorted(results, key=lambda x: x["id"])
    elif process_num > 1:
        chunk_size = len(questions) // process_num
        if len(questions) % process_num == 0:
            with multiprocessing.Pool(processes=process_num) as pool:
                results = [
                    pool.apply_async(
                        inference,
                        args=(
                            model_name,
                            questions[
                                i
                                * chunk_size : min((i + 1) * chunk_size, len(questions))
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
                gathered_responses = [
                    item for sublist in gathered_responses for item in sublist
                ]
                gathered_responses = sorted(gathered_responses, key=lambda x: x["id"])
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
                        inference,
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
                gathered_responses = sorted(gathered_responses, key=lambda x: x["id"])
                return gathered_responses


def format_examples(message, **kwargs) -> str:
    assert "examples" in message
    task_generation_prompt = DEFAULT_TASK_GENERATION_PROMPT
    if len(kwargs["instruction_pool"]) < 2:
        rand_examples_idxs = random.sample(range(0, len(kwargs["ift_dataset"])), k=2)
        for idx in rand_examples_idxs:
            message["examples"].append(
                kwargs["ift_dataset"][idx]["conversations"][0]["value"]
            )
    else:
        message["examples"].extend(random.sample(kwargs["instruction_pool"], k=2))
    assert len(message["examples"]) == 8
    random.shuffle(message["examples"])
    for idx, instruction in enumerate(message["examples"]):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        task_generation_prompt += f"{idx+1}. {instruction}\n"
    task_generation_prompt += f"{9}."
    return task_generation_prompt


def extract_prompts(answer, **kwargs) -> str:
    prompts = post_process_instructs([answer])
    if "instruction_pool" in kwargs:
        kwargs["instruction_pool"].extend(prompts)
    return prompts


def format_conversation(conv, **kwargs) -> str:
    template_name = kwargs.get("template_name")
    conversation = Conversation.from_template(template_name)
    conversation.fill_in_messages(conv)
    return conversation.get_conversation_str(add_generation_prompt=True)


def format_prompt(pair, **kwargs) -> str:
    prompt_reward_model = META_JUDGE_EN.format(
        instruction=pair["instruct"], response=pair["response"]
    )
    template_name = kwargs.get("template_name")
    conversation = Conversation.from_template(template_name)
    conversation.fill_in_messages(
        {"conversations": [{"from": "human", "value": prompt_reward_model}]}
    )
    return conversation.get_conversation_str(add_generation_prompt=True)


def parse_reward_fn(llm_response: str, **kwargs) -> float:
    reward_regex_str = kwargs.get("reward_regex_str")
    result = re.search(rf"{reward_regex_str}", llm_response)
    if result is None:
        return None
    try:
        result.group(1)
    except Exception as e:
        print("Encountered {} while parsing reward function.".format(e))
        return None
    return float(result.group(1))


class DPODatasetGenerator:
    def __init__(
        self,
        model_name,
        template_name,
        backend,
        num_gpus_per_model,
    ):
        self.model_name = model_name
        self.template_name = template_name
        self.backend = backend
        self.num_gpus_per_model = num_gpus_per_model

    def generate_responses(self, prompts, repeat_time=4):
        prompts = sorted(prompts, key=lambda x: x["id"])
        responses = parallel_inference(
            model_name=self.model_name,
            questions=prompts,
            num_choices=repeat_time,
            input_hook_fn=format_conversation,
            output_hook_fn=identity,
            num_gpus_per_model=self.num_gpus_per_model,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=512,
            template_name=self.template_name,
            backend=self.backend,
        )
        prompt_response_pairs = []
        prompts = sorted(prompts, key=lambda x: x["id"])
        for prompt, response in zip(prompts, responses):
            assert prompt["id"] == response["id"]
            for choice in response["choices"]:
                prompt_response_pairs.append(
                    {
                        "id": str(prompt["id"] * repeat_time) + str(choice["id"]),
                        "instruct": prompt["conversations"][0]["value"],
                        "response": choice["output"],
                    }
                )
        return prompt_response_pairs

    def generate_scores(self, instr_resp_pairs, score_time=3):

        reward_regex = create_parse_reward_fn(DEFAULT_REWARD_REGEX_TEMPLATE)
        raw_scores = parallel_inference(
            model_name=self.model_name,
            questions=instr_resp_pairs,
            num_choices=score_time,
            input_hook_fn=format_prompt,
            output_hook_fn=parse_reward_fn,
            num_gpus_per_model=self.num_gpus_per_model,
            temperature=0.7,
            top_p=0.9,
            max_length=4096,
            max_new_tokens=512,
            reward_regex_str=reward_regex,
            template_name=self.template_name,
            backend=self.backend,
        )

        instr_resp_pairs = sorted(instr_resp_pairs, key=lambda x: x["id"])
        conv_score = []
        for pair, score in zip(instr_resp_pairs, raw_scores):
            assert pair["id"] == score["id"]
            assert len(score["choices"]) == score_time
            new_pair = deepcopy(pair)
            new_pair["score"] = score
            conv_score.append(new_pair)

        return conv_score

    def generate_preferences(self, conv_scores, repeat_time=4):
        formed_data = []
        for i in range(len(conv_scores)):
            if i % repeat_time == 0:
                formed_data.append(
                    {
                        "id": i // repeat_time,
                        "instruct": conv_scores[i]["instruct"],
                        "responses": [],
                        "scores": [],
                    }
                )
            score = [
                s["output"]
                for s in conv_scores[i]["score"]["choices"]
                if s["output"] is not None and s["output"] >= 0 and s["output"] <= 5
            ]
            if len(score) > 0:
                formed_data[-1]["scores"].append(sum(score) / len(score))
                formed_data[-1]["responses"].append(conv_scores[i]["response"])

        preference_data = []
        for data in formed_data:
            assert len(data["scores"]) == len(data["responses"])
            if len(data["scores"]) < 2:
                continue
            sorted_index = np.argsort(np.array(data["scores"]))
            if data["scores"][sorted_index[0]] < data["scores"][sorted_index[-1]]:
                preference_data.append(
                    {
                        "prompt": data["instruct"],
                        "prompt_id": len(preference_data),
                        "chosen": [
                            {"value": data["instruct"], "from": "human"},
                            {
                                "value": data["responses"][sorted_index[-1]],
                                "from": "gpt",
                            },
                        ],
                        "rejected": [
                            {"value": data["instruct"], "from": "human"},
                            {
                                "value": data["responses"][sorted_index[0]],
                                "from": "gpt",
                            },
                        ],
                        "conversations": [
                            {"value": data["instruct"], "from": "human"},
                            {
                                "value": data["responses"][sorted_index[-1]],
                                "from": "gpt",
                            },
                        ],
                        "score_chosen": data["scores"][sorted_index[-1]],
                        "score_rejected": data["scores"][sorted_index[0]],
                    }
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
        "--template-name",
        type=str,
        required=True,
        help="The template name.",
    )
    parser.add_argument(
        "--instruction-path",
        type=str,
        required=True,
        help="The path to the the seed data. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--sft-base-model",
        type=str,
        choices=["ift", "eft"],
        required=True,
        help="The optimized sft baseline.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        required=True,
        help="The optimized sft baseline.",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        required=True,
        help="A debug option. The end index of questions.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-scores",
        type=int,
        default=3,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="The path to the the output data.",
    )

    args = parser.parse_args()

    generator = DPODatasetGenerator(
        model_name=args.model_path,
        template_name=args.template_name,
        backend=args.backend,
        num_gpus_per_model=args.num_gpus_per_model,
    )

    filtered_prompt = load_json(f"{str(args.instruction_path)}")
    print(f"Data Length: {len(filtered_prompt)}")
    prompt_response_pairs = generator.generate_responses(prompts=filtered_prompt)
    dump_json(
        prompt_response_pairs,
        f"{args.output_path}/prompts_responses.json",
        indent=2,
    )

    conv_scores = generator.generate_scores(instr_resp_pairs=prompt_response_pairs)
    dump_json(
        conv_scores,
        f"{args.output_path}/conv_scores.json",
        indent=2,
    )
    # scores = json.load(open("data/{}/conv_scores.json".format(args.model_id), 'r'))
    preference_data = generator.generate_preferences(conv_scores=conv_scores)
    dump_json(
        preference_data,
        f"{args.output_path}/preference_data.json",
        indent=2,
    )