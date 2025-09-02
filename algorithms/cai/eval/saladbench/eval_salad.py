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
from autoalign.inference.inferencer import (
    MultiProcessHFInferencer,
    MultiProcessVllmInferencer,
)

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