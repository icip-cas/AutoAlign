from datasets import load_dataset
from vllm import LLM, SamplingParams
from saladbench.evaluator import MDJudgeEvaluator ,DDPMDJudgeEvaluator
from transformers import AutoTokenizer
from typing import List
import ray
import torch
import math
import socket
import json
import argparse
import os

class MultiProcessVllmInferencer:
    def __init__(
        self,
        model_path: str,
        num_gpus_per_model: int = 1,
        do_sample: bool = False,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty=0.0,
        stop: List[str] = [],
    ):

        self.num_gpus_total = torch.cuda.device_count()
        self.num_gpus_per_model = num_gpus_per_model

        self.model_path = model_path

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            use_beam_search=(not do_sample) and (not num_beams == 1),
            best_of=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=1.0,
            frequency_penalty=frequency_penalty,
            stop=stop,
            early_stopping=False,
        )

        self.use_ray = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.num_gpus_total // num_gpus_per_model > 1:
            self.use_ray = True
            ray.init(ignore_reinit_error=True)

    def find_n_free_ports(self, n):
        ports = []
        sockets = []
        port_range_start = 5000
        port_range_end = 8000
        current_port = port_range_start

        while len(ports) < n and current_port <= port_range_end:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("", current_port))
                ports.append(current_port)
                sockets.append(s)
            except OSError:
                # 如果端口已经被占用，继续尝试下一个端口
                pass
            current_port += 1

        if len(ports) < n:
            raise RuntimeError(
                f"Could only find {len(ports)} free ports within the specified range."
            )

        return ports, sockets

    @staticmethod
    def single_process_inference(
        model_path, num_gpus_per_model, vllm_port, *args, **kwargs
    ):

        # Set an available port
        os.environ["VLLM_PORT"] = str(vllm_port)
        print(f"Using VLLM_PORT: {vllm_port}")

        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus_per_model,
            enforce_eager=True,
            trust_remote_code=True,
        )

        return model.generate(*args, **kwargs)

    def inference(self, data: List[str]):
        if self.use_ray:
            get_answers_func = ray.remote(num_gpus=self.num_gpus_per_model)(
                MultiProcessVllmInferencer.single_process_inference
            ).remote
        else:
            get_answers_func = MultiProcessVllmInferencer.single_process_inference

        num_processes = min(
            len(data), max(1, self.num_gpus_total // self.num_gpus_per_model)
        )
        chunk_size = math.ceil(len(data) / num_processes)

        ports, sockets = self.find_n_free_ports(num_processes)

        gathered_responses = []
        for idx, i in enumerate(range(0, len(data), chunk_size)):
            gathered_responses.append(
                get_answers_func(
                    self.model_path,
                    self.num_gpus_per_model,
                    ports[idx],
                    data[i : i + chunk_size],
                    self.sampling_params
                )
            )

        for s in sockets:
            s.close()

        if self.use_ray:
            gathered_responses = ray.get(gathered_responses)

        gathered_responses = [
            item for sublist in gathered_responses for item in sublist
        ]
        gathered_responses = [
            response.outputs[0].text for response in gathered_responses
        ]
        
        torch.cuda.empty_cache()

        return gathered_responses

    def get_tokenizer(self):

        return self.tokenizer

    def __del__(self):
        if self.use_ray:
            ray.shutdown()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model safety on Salad dataset')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the model or model name')
    parser.add_argument('--outname', type=str, required=True,
                      help='Output file name for results')
    return parser.parse_args()

def load_and_process_data(model_path):
    # Load dataset
    dataset = load_dataset("OpenSafetyLab/Salad-Data", name='base_set', split='train')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Process conversations
    prompts = []
    questions = []
    for item in dataset:
        messages = [
            {"role": "user", "content": item["question"]}
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted)
        questions.append(item["question"])
    
    return questions,prompts

def generate_responses(prompts, model_path):
    # Initialize LLM
    inferencer = MultiProcessVllmInferencer(
    model_path=model_path,
    num_gpus_per_model=1,
    do_sample=True,
    num_beams=1,
    max_new_tokens=1024,
    temperature=0.8,
    top_p=0.95
    )

    outputs = inferencer.inference(prompts)
    # Extract prompts and responses
    questions = []
    answers = []
    
    for output, prompt in zip(outputs, prompts):
        questions.append(prompt)
        answers.append(output)
        
    return questions, answers

def evaluate_safety(questions, answers):
    # Initialize safety evaluator
    model_ckpt = "MD-Judge-v0.1"
    md_judge = DDPMDJudgeEvaluator(
        model_ckpt, 
        use_vllm=True,
        use_flash_attn=True
    )
    eval_labels, eval_clses = md_judge.eval_unsafe_labels(questions, answers)
    return eval_labels, eval_clses

def save_results(questions, answers, eval_labels, eval_clses, outname):
    results = {
        "evaluations": [
            {
                "question": q,
                "answer": a,
                "safety_label": ["SAFE", "UNSAFE", "Exception"][l],
                "harm_categories": c
            }
            for q, a, l, c in zip(questions, answers, eval_labels, eval_clses)
        ]
    }
    
    with open(outname, "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Process the dataset
    print("Processing dataset...")
    raw_questions, prompts = load_and_process_data(args.model)
    
    # Generate responses
    print(f"Generating responses using model: {args.model}")
    questions, answers = generate_responses(prompts, args.model)
    
    # Evaluate safety
    print("Evaluating safety...")
    eval_labels, eval_clses = evaluate_safety(raw_questions, answers)
    
    # Save results
    print(f"Saving results to {args.outname}...")
    save_results(raw_questions, answers, eval_labels, eval_clses, args.outname)
    print(f"Evaluation completed. Results saved to '{args.outname}'")
    
    # 统计labels
    print("统计labels")
    labels = ["SAFE", "UNSAFE", "Exception"]
    label_counts = [eval_labels.count(i) for i in range(len(labels))]
    print("统计结果:")
    for i in range(len(labels)):
        print(f"{labels[i]}: {label_counts[i]}")

if __name__ == "__main__":
    main()