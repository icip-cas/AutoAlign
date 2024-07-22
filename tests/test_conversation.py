"""test conversation"""
import json
from itertools import groupby
import pytest
from tqdm import tqdm
from transformers import AutoTokenizer
from autoalign.conversation import Role, Conversation, TEMPLATES, IGNORED_TOKEN_ID

@pytest.fixture
def dummy_data():
    with open('./data/dummy_sft.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@pytest.fixture(params=TEMPLATES.keys())
def template_conversation(request):
    template_name = request.param
    print(f"Testing template: {template_name}")
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

def test_invalid_template():
    with pytest.raises(ValueError):
        Conversation.from_template("non_existent_template")

# def test_get_conversation_str(vicuna_conversation):
#     vicuna_conversation.append_message(Role.HUMAN, "Hello")
#     vicuna_conversation.append_message(Role.ASSISTANT, "Hi there!")
#     expected = "USER: Hello ASSISTANT: Hi there!</s>"
#     assert vicuna_conversation.get_conversation_str() == expected

# def test_llama2_strategy(llama2_conversation):
#     llama2_conversation.append_message(Role.HUMAN, "Hello")
#     llama2_conversation.append_message(Role.ASSISTANT, "Hi there!")
#     expected = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nHello [/INST] Hi there! </s>"
#     assert llama2_conversation.get_conversation_str() == expected

# def test_conversation_str_after_fill(vicuna_conversation, dummy_data):
#     english_conv = dummy_data[0]
#     vicuna_conversation.fill_in_messages(english_conv)
    
#     expected_str = "USER: Tell me about Beethoven. ASSISTANT: Beethoven is a great composer.</s>USER: When was he born? ASSISTANT: He was born in 1770.</s>"
#     assert vicuna_conversation.get_conversation_str() == expected_str

# def test_fill_in_messages_keep_system(dummy_data):
#     conv = Conversation.from_template("llama-2-chat-keep-system")
#     english_conv = dummy_data[0]
#     conv.fill_in_messages(english_conv)
    
#     assert conv.system_message == "You are a helpful artificial assistant who gives friendly responses."
#     assert len(conv.messages) == 5  # 1 system + 2 human + 2 gpt
    
#     expected_str = "<s>[INST] <<SYS>>\nYou are a helpful artificial assistant who gives friendly responses.\n<</SYS>>\n\nTell me about Beethoven. [/INST] Beethoven is a great composer. </s><s>[INST] When was he born? [/INST] He was born in 1770.</s>"
#     assert conv.get_conversation_str() == expected_str

# def test_fill_in_messages_invalid_role(vicuna_conversation):
#     invalid_conv = {
#         "conversations": [
#             {"from": "invalid_role", "value": "This is an invalid role."}
#         ]
#     }
#     with pytest.raises(ValueError, match="Invalid role: invalid_role"):
#         vicuna_conversation.fill_in_messages(invalid_conv)

@pytest.mark.parametrize("template_name, tokenizer_name_or_path", [
    ("llama-2-chat", "pretrained_models/Llama-2-7b-chat-hf"),
    ("llama-3-instruct", "/data7/hf_models/NousResearch/Meta-Llama-3-8B"),
    ("chatml", "pretrained_models/Qwen2-7B"),
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
            assert target_text.strip() == assistant_response.strip(), "target text and gpt response do not match!"