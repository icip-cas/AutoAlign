import torch
import random
import re
import multiprocessing
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from datasets import Dataset
from copy import deepcopy
from rouge_score import rouge_scorer
import argparse
import os
from utils import (
    load_json,
    dump_json,
    get_available_gpu_list,
    post_process_instructs,
    DEFAULT_TASK_GENERATION_PROMPT,
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
            assert "index" in question
            choices = []
            prompt = input_hook_fn(question, **kwargs)
            for i in range(num_choices):
                torch.manual_seed(i)
                try:
                    output = inferencer.inference(prompt, **infer_kwargs)
                    output = output_hook_fn(output, **kwargs)
                except RuntimeError as e:
                    print(e)
                    print("ERROR question ID: ", question["index"])
                    output = "ERROR"
                choices.append({"index": i, "output": output})
            answers.append({"index": question["index"], "choices": choices})
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
            assert "index" in question
            choices = []
            prompt = input_hook_fn(question, **kwargs)
            outputs = llm.generate(prompt, sampling_params)
            for output in outputs:
                prompt = output.prompt
                for idx, generated_output in enumerate(output.outputs):
                    generated_text = generated_output.text
                    completion = output_hook_fn(generated_text, **kwargs)
                    choices.append({"index": idx, "output": completion})
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
    try:
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    except Exception as e:
        print(e)
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
        return sorted(results, key=lambda x: x["index"])
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
                gathered_responses = sorted(
                    gathered_responses, key=lambda x: x["index"]
                )
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


class DatasetGenerator:
    def __init__(
        self,
        instruct_generator,
        template_name,
        backend,
        ift_dataset,
        num_gpus_per_model,
    ):
        self.instruct_generator = instruct_generator
        self.template_name = template_name
        self.backend = backend
        self.ift_dataset = Dataset.from_list(load_json(ift_dataset))
        self.num_gpus_per_model = num_gpus_per_model

    def generate_prompts(self, num_prompts: int, num_examples: int = 6):
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
            model_name=self.instruct_generator,
            questions=task_generation_prompts,
            num_choices=1,
            input_hook_fn=format_examples,
            output_hook_fn=extract_prompts,
            num_gpus_per_model=self.num_gpus_per_model,
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=512,
            ift_dataset=self.ift_dataset,
            instruction_pool=[],
            template_name=self.template_name,
            backend=self.backend,
        )
        task_prompts = []
        for raw_prompt in raw_task_prompts:
            task_prompts.extend(raw_prompt["choices"][0]["output"])

        rem_dup_task_prompts = list(set(task_prompts))
        new_prompts = post_process_instructs(rem_dup_task_prompts)
        return new_prompts

    def prompts_filter(self, prompts: list[dict]):
        os.environ["TOKENIZERS_PARALLELISM"] = "True"
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        assert len(prompts) > 0
        random.shuffle(prompts)
        filtered_prompts = [prompts[0]]
        for prompt in tqdm(prompts[1:]):
            with Pool(64) as p:
                rouge_scores = p.map(partial(scorer.score, prompt), filtered_prompts)
            rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
            if max(rouge_scores) > 0.7:
                continue
            else:
                filtered_prompts.append(prompt)
        return [
            {"index": index, "conversations": [{"from": "human", "value": prompt}]}
            for index, prompt in enumerate(filtered_prompts)
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--question-gen-model-path",
        type=str,
        required=True,
        help="The path to the question generator (usually an instruct model) weights. \
        This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--seed-data-path",
        type=str,
        required=True,
        help="The path to the the seed data. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        required=True,
        help="The optimized sft baseline.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="The times to sample from the model.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output",
        help="The times to sample from the model.",
    )
    args = parser.parse_args()

    generator = DatasetGenerator(
        instruct_generator=args.question_gen_model_path,
        ift_dataset=args.seed_data_path,
        template_name=args.template_name,
        backend=args.backend,
        num_gpus_per_model=args.num_gpus_per_model,
    )

    raw_prompts = generator.generate_prompts(num_prompts=args.num_prompts)
    filtered_prompt = generator.prompts_filter(prompts=raw_prompts)

    dump_json(
        filtered_prompt,
        os.path.join(
            args.output_path,
            "{}_filtered_prompts.json".format(
                args.model_id,
            ),
        ),
        indent=2,
    )
