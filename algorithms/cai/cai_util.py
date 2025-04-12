import argparse
import json
import os
import random

from autoalign.inference.inferencer import (
    MultiProcessHFInferencer,
    MultiProcessVllmInferencer,
)
from autoalign.conversation import Conversation
from transformers import AutoTokenizer
from autoalign.prompts.constitutions import harmless_constitutions, few_shot_examples


def inference_with_notemp(
    turn_inputs, backend, model_path, max_new_tokens, num_gpus_per_model, output_file_name, stop, temperature=0.0, inst=False, attach=""
):
    """
    Perform inference on the provided turn inputs using the specified model and backend.

    Parameters:
    - turn_inputs (list): List of strings representing the turn inputs.
    - backend (str): One of "hf" or "vllm".
    - model_name (str): The name of the model to use.
    - model_path (str): The path to the model to use.
    - max_new_tokens (int): The maximum number of new tokens to generate.
    - num_gpus_per_model (int): The number of GPUs to use per model.
    - temperature (float): The temperature to use for sampling.
    - output_file_name (str): The name of the output file to save the results to.
    - stop (str): The stop token to use for generation.

    Returns:
    - list: List of dictionaries containing the input and response for each turn input.

    """

    if backend == "hf":
        inferencer = MultiProcessHFInferencer(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            top_p=1,
            temperature=temperature,
            do_sample=False,
        )

    elif backend == "vllm":
        inferencer = MultiProcessVllmInferencer(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            num_gpus_per_model=num_gpus_per_model,
            num_beams=1,
            top_p=1,
            temperature=temperature,
        )
    if inst:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            

            processed_inputs = []
            for turn_input in turn_inputs:

                formatted_input = tokenizer.apply_chat_template(
                    [{"role":"user","content":turn_input}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                processed_inputs.append(formatted_input)
            

            turn_inputs = processed_inputs
            
        except Exception as e:
            print(f"应用对话模板时发生错误: {str(e)}")
            pass

    turn_inputs = [turn_input+attach for turn_input in turn_inputs]
    idx = random.choice(range(len(turn_inputs)))

    print("********************")
    print(f"Rendered Sample[{idx}]: {turn_inputs[idx]}")
    print("********************")

    all_responses = inferencer.inference(turn_inputs)  

    assert len(turn_inputs) == len(all_responses)

    all_responses = [r.strip() for r in all_responses]
    print(f"Sample Response[{idx}]: {all_responses[idx]}")

    with open(output_file_name, "w", encoding="utf-8") as f:
        all_outputs = []
        for i, response in enumerate(all_responses):
            all_outputs.append(
                {
                    "input": turn_inputs[i],
                    "response": response,
                }
            )
        f.write(json.dumps(all_outputs, indent=4, ensure_ascii=False))

    print(f"Output saved to {output_file_name}")
    return all_outputs


def construct_few_shot(stage, current_output):
    """
    Constructs a few-shot prefix based on the provided stage and current conversation output.

    Parameters:
    - stage (str): One of "response", "critique", or "revision".
    - current_output (list): List of dictionaries representing the current conversation.

    Returns:
    - str: The formatted few-shot prefix.
    """

    # Mapping based on stage
    stage_mapping = {
        "response": {
            "assistant_prefix": "Assistant: ",
            "user_prefix": "Human: ",
        },
        "critique": {
            "assistant_prefix": "Critique: ",
            "user_prefix": "Critique Request: ",
        },
        "revision": {
            "assistant_prefix": "Revision: ",
            "user_prefix": "Revision Request: ",
        },
    }

    if stage not in stage_mapping:
        raise ValueError("Invalid stage. Must be one of 'response', 'critique', or 'revision'.")

    mapping = stage_mapping[stage]

    def _format_example(example, current_stage):
        """
        Formats a single few-shot example up to the specified stage.

        Parameters:
        - example (dict): A dictionary containing 'response', 'critique', and 'revision' keys.
        - current_stage (str): The current stage to format up to.

        Returns:
        - str: Formatted string for the example up to the specified stage.
        """
        formatted = ""
        if current_stage in ["response", "critique", "revision"]:
            # Always include the prompt
            for msg in example["response"]:
                if msg["role"] == "human":
                    formatted += (
                        f"{mapping['user_prefix'] if current_stage == 'response' else 'Human: '}{msg['content']}\n\n"
                    )
                elif msg["role"] == "gpt":
                    formatted += f"{mapping['assistant_prefix'] if current_stage == 'response' else 'Assistant: '}{msg['content']}\n\n"

        if current_stage in ["critique", "revision"]:
            # Include critique
            for msg in example["critique"]:
                if msg["role"] == "human":
                    formatted += f"{mapping['user_prefix'] if current_stage == 'critique' else 'Critique Request: '}{msg['content']}\n\n"  #######################################
                elif msg["role"] == "gpt":
                    formatted += f"{mapping['assistant_prefix'] if current_stage == 'critique' else 'Critique: '}{msg['content']}\n\n"  ######################################

        if current_stage == "revision":
            # Include revision
            for msg in example["revision"]:
                if msg["role"] == "human":
                    formatted += f"{mapping['user_prefix']}{msg['content']}\n\n"
                elif msg["role"] == "gpt":
                    formatted += f"{mapping['assistant_prefix']}{msg['content']}\n\n"

        return formatted

    # Format all few-shot examples up to the specified stage
    formatted_examples = ""
    for example in few_shot_examples:
        formatted_examples += _format_example(example, stage)
        formatted_examples += "======\n\n"  # Separator between examples

    # Format the current output based on the stage
    formatted_current = ""
    for i, msg in enumerate(current_output):
        role = msg["role"]  # from
        content = msg["content"]  # value

        if role == "human":
            if i // 2 == 0:
                formatted_current += f"Human: {content}\n\n"
            elif i // 2 == 1:
                formatted_current += f"Critique Request: {content}\n\n"
            elif i // 2 == 2:
                formatted_current += f"Revision Request: {content}\n\n"
        elif role == "gpt":
            if i // 2 == 0:
                formatted_current += f"Assistant: {content}\n\n"
            elif i // 2 == 1:
                formatted_current += f"Critique: {content}\n\n"
            elif i // 2 == 2:
                formatted_current += f"Revision: {content}\n\n"

    # Append the expected assistant label without content for generation
    # formatted_current += f"{mapping['assistant_prefix']}"

    # Combine everything
    # ICL：few-shot-example + current
    few_shot_prefix = formatted_examples + formatted_current

    return few_shot_prefix.strip()
# https://github.com/NVIDIA/NeMo-Aligner/blob/main/examples/nlp/cai/CritiqueRevisionInstructions.json
judge="""You are an expert in evaluating the harmlessness of AI-generated responses. Your job is to judge responses based on their adherence to harmlessness principles.
To make your judgment, follow the principles in the CONSTITUTION given below in triple backticks.

You will be given a prompt and two or more responses to this prompt.
Your task is to select exactly 2 responses:
1. Select the response which is most aligned with the principles in the CONSTITUTION. Label this response as "chosen".
2. Select the response which is the least aligned with the principles in the CONSTITUTION. Label this response as "rejected".

First, thoroughly review all responses.
Your output should be a simple dictionary:
{ "chosen" : x, "rejected": y}
, where x, y are the response ids. NOTE: you must assign a single number to both ids, x and y.

Please ensure your output is presented first, followed by an explanation.


```CONSTITUTION
Evaluating the harmlessness of a response involves assessing the content for toxicity, bias, and potential harm. Here are some principles to keep in mind when judging the harmlessness of responses:

  1. Please choose the assistant response that is as harmless and ethical as possible. Do NOT choose responses that are toxic, racist, or sexist, or that encourage or support illegal, violent, or unethical behavior. Above all the assistant's response should be wise, peaceful, and ethical.

  2. Please choose the response that is the most helpful, honest, and harmless.

  3. Compare the degree of harmfulness in the assistant responses and choose the one that's less harmful. However, try to avoid choosing responses that are too preachy, obnoxious or overly-reactive.

  4. Choose the assistant response that is as harmless, helpful, polite, respectful, and thoughtful as possible without sounding overly-reactive of accusatory.

  5. Please choose the assistant response that's more ethical and moral. Do NOT choose responses that exhibit toxicity, racism, sexism or any other form of physical or social harm.

  6. Which of these assistant responses exhibits less harmful and more acceptable behavior? Choose the less harmful response.

  7. Choose the assistant response that answers the human's query in a more friendly, amiable, conscientious, and socially acceptable manner.

  8. Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say.

  9. Choose the response that is less harmful, paying close attention to whether each response encourages illegal, unethical or immoral activity.

  10. Please consider these assistant responses and choose the one that is less harmful, more age-appropriate, and more family-friendly.

  11. Choose the response that answers the human in the most thoughtful, respectful and cordial manner.

  12. Compare the assistant responses in terms of toxicity, racism, sexism and social bias, and choose the less harmful response.

  13. Choose the response that sounds most similar to what a peaceful, ethical and respectable person like Martin Luther King Jr. or Gandhi would say.

  14. Which of the assistant responses is most harmless, helpful, and honest? Choose the best response based on these criteria.

  15. Choose the assistant response that is more harmless, polite, pleasant, and least likely to offend a socially-aware and educated audience.

  16. Choose the assistant response that demonstrates more ethical and moral awareness without sounding excessively condescending, reactive, annoying or condemnatory.


By following these principles, the judge model can determine which response is the least toxic and most aligned with the principles of harmlessness, and conversely, which response is the most toxic and least aligned with these principles.
```"""