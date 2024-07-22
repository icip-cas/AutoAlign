"""test conversation"""
import json
from itertools import groupby
import pytest
from tqdm import tqdm
from transformers import AutoTokenizer
from autoalign.conversation import Role, Conversation, TEMPLATES, IGNORED_TOKEN_ID

def approx_equal(str1, str2): return ''.join(str1.split()) == ''.join(str2.split())

@pytest.fixture
def dummy_data():
    with open('./data/dummy_sft.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@pytest.fixture(params=[template_name for template_name in TEMPLATES.keys() if template_name != 'gpt-4'])
def template_conversation(request):
    template_name = request.param
    return Conversation.from_template(template_name)

def test_append_message(template_conversation):
    template_conversation.append_message(Role.HUMAN, "Hello")
    assert template_conversation.messages[-1] == (Role.HUMAN, "Hello")

def test_update_last_message(template_conversation):
    template_conversation.append_message(Role.HUMAN, "Hello")
    template_conversation.update_last_message("Hi there")
    assert template_conversation.messages[-1] == (Role.HUMAN, "Hi there")

def test_fill_in_messages(template_conversation):
    conv_dict = {
        "conversations": [
            {"from": "system", "value": "You are a helpful assistant."},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there!"}
        ]
    }
    template_conversation.fill_in_messages(conv_dict)
    assert template_conversation.system_message == "You are a helpful assistant."
    assert template_conversation.messages[1:] == [(Role.HUMAN, "Hello"), (Role.ASSISTANT, "Hi there!")]

def test_to_openai_api_messages():

    conv = Conversation.from_template('gpt-4')

    print(conv.to_openai_api_messages())
    
    assert conv.to_openai_api_messages() == []
    
    conv.system_message = "You are a helpful assistant."
    expected = [{"role": "system", "content": "You are a helpful assistant."}]
    assert conv.to_openai_api_messages() == expected
    
    conv.append_message(Role.HUMAN, "Hello, how are you?")
    expected.append({"role": "user", "content": "Hello, how are you?"})
    assert conv.to_openai_api_messages() == expected
    
    conv.append_message(Role.ASSISTANT, "I'm doing well, thank you for asking. How can I assist you today?")
    expected.append({"role": "assistant", "content": "I'm doing well, thank you for asking. How can I assist you today?"})
    assert conv.to_openai_api_messages() == expected
    
    conv.append_message(Role.HUMAN, "What's the weather like today?")
    expected.append({"role": "user", "content": "What's the weather like today?"})
    assert conv.to_openai_api_messages() == expected
    
    conv = Conversation.from_template('gpt-4')
    conv.append_message(Role.HUMAN, "Hi")
    conv.append_message(Role.ASSISTANT, "Hello!")
    assert conv.to_openai_api_messages() == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}
    ]

def test_invalid_template():
    with pytest.raises(ValueError):
        Conversation.from_template("non_existent_template")

@pytest.mark.parametrize("template_name, tokenizer_name_or_path", [
    ("llama-2-chat", "pretrained_models/Llama-2-7b-chat-hf"),
    ("llama-2-chat-keep-system", "pretrained_models/Llama-2-7b-chat-hf"),
    ("llama-3-instruct", "/data7/hf_models/NousResearch/Meta-Llama-3-8B"),
    ("chatml", "pretrained_models/Qwen2-7B"),
    ("chatml-keep-system", "pretrained_models/Qwen2-7B"),
])
def test_get_tokenized_conversation(
    template_name: str,
    tokenizer_name_or_path: str,
    dummy_data,
    model_max_length: int = 4096,
):
    """test tokenization and labels preparing"""
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    # test on each data point
    for conv in tqdm(dummy_data):
        # get conversation from template
        conversation = Conversation.from_template(template_name)
        # fill in messages from conv
        conversation.fill_in_messages(conv)
        # tokenize conversation
        tokenized_conversation = conversation.get_tokenized_conversation(
            tokenizer=tokenizer,
            model_max_length=model_max_length,
        )
        input_ids = tokenized_conversation["input_ids"]
        labels = tokenized_conversation["labels"]
        
        # check whether ids match
        assert len(input_ids) == len(labels), "length of input_ids and labels do not match!"
        for input_id, label in zip(input_ids, labels):
            assert label == IGNORED_TOKEN_ID or input_id == label, "valid token id in input_ids and labels do not match!"
        
        # check whether target texts match
        target_ids = [list(y) for x, y in groupby(labels, lambda x: x != IGNORED_TOKEN_ID) if x]
        target_texts = list(map(tokenizer.decode, target_ids))
        print(" ".join(target_texts))
        
        assistant_responses = [_["value"] for _ in conv["conversations"] if _["from"] == "gpt"]
        for target_text, assistant_response in zip(target_texts, assistant_responses):
            assert approx_equal(
                target_text.strip(),
                assistant_response.strip() + conversation.template.role_ends[Role.ASSISTANT]
            ), "target text and gpt response do not match!"