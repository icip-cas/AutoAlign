from transformers import AutoTokenizer
import json
from autoalign.conversation import (
    Role,
    Conversation,
)
import subprocess
import pytest
from autoalign.eval.run_eval import generate_config


def approx_equal(str1, str2):
    return "".join(str1.split()) == "".join(str2.split())


@pytest.fixture
def test_data_from():
    with open("./data/gsm8k_example_from.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def test_data_role():
    with open("./data/gsm8k_example_role.json", "r", encoding="utf-8") as f:
        return json.load(f)


def opencompass_prompt(config_path: str, opencompass_path: str):
    command = [
        " cd {opencompass_path} \n python tools/prompt_viewer.py {config_path} -n".format(
            config_path=config_path,
            opencompass_path=opencompass_path,
        )
    ]
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, shell=True)
        # wait
        output = process.stdout.decode("utf-8")
        return output.split(
            "----------------------------------------------------------------------------------------------------"
        )[-2][1:-1]
    except Exception as e:
        print(f"Opencompass excecuted with errors: {e}")


def conversation_prompt(template_name: str, test_data_from):
    assert len(test_data_from) == 1
    conv = test_data_from[0]
    # get conversation from template
    conversation = Conversation.from_template(template_name)
    # fill in messages from conv
    conversation.fill_in_messages(conv)
    if conversation.messages[0][0] == Role.SYSTEM:
        conversation.pop_message(0)
    return conversation.get_conversation_str(add_generation_prompt=True)


def tokenized_conversation_prompt(
    tokenizer_name_or_path: str, template_name: str, test_data_from
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    assert len(test_data_from) == 1
    conv = test_data_from[0]
    # get conversation from template
    conversation = Conversation.from_template(template_name)
    # fill in messages from conv
    conversation.fill_in_messages(conv)
    if conversation.messages[0][0] == Role.SYSTEM:
        conversation.pop_message(0)
    if "chatml" in template_name:
        conversation.messages.insert(0, (Role.SYSTEM, "You are a helpful assistant."))
    return tokenizer.decode(
        tokenizer.encode(conversation.get_conversation_str(add_generation_prompt=True))
    )


def tokenizer_prompt(tokenizer_name_or_path: str, test_data_role):
    """test tokenizer prompting"""
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    assert len(test_data_role) == 1
    conv = test_data_role[0]
    tokenized = tokenizer.apply_chat_template(
        conv["conversations"], add_generation_prompt=True
    )
    return tokenizer.decode(tokenized)


@pytest.mark.parametrize(
    "template_name, tokenizer_name_or_path",
    [
        ("llama-2-chat", "pretrained_models/llama-2/llama-2-7b-chat-hf"),
        ("llama-2-chat-keep-system", "pretrained_models/llama-2/llama-2-7b-chat-hf"),
        ("vicuna_v1.1", "pretrained_models/llama-2/llama-2-7b-chat-hf"),
        ("llama-3-instruct", "pretrained_models/NousResearch/Meta-Llama-3-8B-Instruct"),
        ("chatml", "pretrained_models/qwen/Qwen2-7B-Instruct"),
        ("chatml-keep-system", "pretrained_models/qwen/Qwen2-7B-Instruct"),
        ("mistral-instruct", "pretrained_models/mistral/Mistral-7B-Instruct-v0.3"),
        ("gemma", "pretrained_models/gemma/gemma-2-9b-it"),
        ("zephyr", "pretrained_models/zephyr-7b-beta"),
    ],
)
def test_opencompass_conversation(
    template_name: str,
    tokenizer_name_or_path: str,
    test_data_from,
):
    # opencompass config
    config_path = generate_config(
        model_name=template_name + "-test",
        model_path=tokenizer_name_or_path,
        eval_type="objective_core",
        per_model_gpu=1,
        batch_size=1,
        opencompass_path="opencompass",
        backend="hf",
        template_name=template_name,
    )
    opencompass_prompt_ = opencompass_prompt(
        config_path=config_path, opencompass_path="opencompass"
    )

    conversation_prompt_ = conversation_prompt(
        template_name=template_name, test_data_from=test_data_from
    )

    assert (
        opencompass_prompt_ == conversation_prompt_
        or opencompass_prompt_ == conversation_prompt_ + " "
    )


@pytest.mark.parametrize(
    "template_name, tokenizer_name_or_path",
    [
        ("llama-3-instruct", "pretrained_models/NousResearch/Meta-Llama-3-8B-Instruct"),
        ("chatml", "pretrained_models/qwen/Qwen2-7B-Instruct"),
        ("chatml-keep-system", "pretrained_models/qwen/Qwen2-7B-Instruct"),
        ("mistral-instruct", "pretrained_models/mistral/Mistral-7B-Instruct-v0.3"),
        ("gemma", "pretrained_models/gemma/gemma-2-9b-it"),
        ("zephyr", "pretrained_models/zephyr-7b-beta"),
    ],
)
def test_get_prompt_conversation(
    template_name: str,
    tokenizer_name_or_path: str,
    test_data_from,
    test_data_role,
):
    _conversation_prompt = tokenized_conversation_prompt(
        template_name=template_name,
        test_data_from=test_data_from,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )

    _tokenizer_prompt = tokenizer_prompt(
        tokenizer_name_or_path=tokenizer_name_or_path, test_data_role=test_data_role
    )

    assert _conversation_prompt == _tokenizer_prompt
