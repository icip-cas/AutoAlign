import openai
from openai import OpenAI
import os

from typing import List, Dict, Any, Optional

from transformers import logging
import torch.distributed as dist

logger = logging.get_logger(__name__)
logging.set_verbosity_info()

def is_rank_0():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

PROMPT = '''You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will receive a question, an output sentence, and the correct answer. Your task is to determine if the output sentence accurately answers the question based on the provided correct answer. Respond with either [Correct] or [Incorrect].
-
Special considerations:

1. **Multiple Answers**: If the output contains multiple answers, evaluate whether later answers modify or correct earlier ones. In such cases, compare the final answer with the correct answer. If the final answer is unclear or incorrect, respond with [Incorrect].

2. **Mathematical Problems**: If the formats differ but the answers are mathematically equivalent, respond with [Correct].

3. **Explicit Options**: If the question provides explicit candidate answers, the output will be considered correct if it clearly indicates the correct option's code or the correct option's content.

4. **No Explicit Options**: If the question does not provide explicit options, the output must align with the correct answer in content and meaning to be considered [Correct].
-

Question: """{question}"""

Output sentence: """{output}"""

Correct answer: {answer}

Judgement:
'''

def xverfiy_reward(prompts: list[list[dict[str, str]]], completions: list[list[dict[str, str]]], answer: list[str], **kwargs) -> list[Optional[float]]:
    """
    This function takes a list of completions and a solution, and returns a list of rewards for each completion.
    The reward is calculated based on the similarity between the completion and the solution.
    """

    base_url = os.environ.get("API_BASE_URL", "http://localhost:8001/v1")
    api_key = os.environ.get("API_KEY", "token_abc")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    prompts = [p[1]["content"] for p in prompts]
    contents = [completion[0]["content"] for completion in completions]

    all_prompts = []

    for p, c in zip(prompts, contents):
        # Extract the question, output, and answer from the completion

        # Create a prompt for the OpenAI API
        prompt = PROMPT.format(
            question=p,
            output=c,
            answer=answer
        )
        
        all_prompts.append(prompt)

    all_responses = []

    for prompt in all_prompts:

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]

        while True:
            # Call the OpenAI API to get the ratings
            try:
                response = client.chat.completions.create(
                    model="default",
                    messages=messages,
                    temperature=0.1,
                    top_p=0.7
                )
            except Exception as e:
                print(f"Error in OpenAI API call: {e}")
                continue

            if response and response.choices and len(response.choices) > 0:
                break
    
        all_responses.append(response)

    # Extract the ratings from the response

    ratings = [r.choices[0].message.content.strip() == "Correct" for r in all_responses]

    # debug
    if is_rank_0():
        for i, r in enumerate(ratings):
            print(f"Prompt: {prompts[i]}", flush=True)
            print(f"Content: {contents[i][-100:]}", flush=True)
            print(f"Answer: {answer[i]}", flush=True)
            print(f"Response: {all_responses[i].choices[0].message.content.strip()}", flush=True)
            print(f"Judge: {r}", flush=True)
            print("-" * 50+"\n")

    rewards = [1. if r else 0. for r in ratings]
    
    # Convert ratings to float and return
    return rewards