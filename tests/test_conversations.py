"""test conversation"""
import json
from itertools import groupby

import fire
from tqdm import tqdm
from transformers import AutoTokenizer

from autoalign.conversation import Conversation, IGNORED_TOKEN_ID


def test_get_tokenized_conversation(
    template_name: str = "qwen-7b-chat",
    tokenizer_name_or_path: str = "Qwen/Qwen1.5-7B",
    model_max_length: int = 4096,
    data_path: str = "data/dummy_conversation.json",
):
    """test tokenization and labels preparing"""
    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # read conv_data
    with open(data_path, "r") as f:
        conv_data = json.load(f)

    # test on each data point
    for conv in tqdm(conv_data):
        # get conversation from template
        conversation = Conversation.from_template(template_name)

        # fill in messages from conv
        conversation.fill_in_messages(conv)

        # tokenize conversation
        tokenized_conversation = conversation.get_tokenized_conversation(
            tokenizer=tokenizer,
            model_max_length=model_max_length,
        )
        input_ids = tokenized_conversation.input_ids
        labels = tokenized_conversation.labels

        # check whether ids match
        assert len(input_ids) == len(labels), "length of input_ids and labels do not match!"
        for input_id, label in zip(input_ids, labels):
            assert label == IGNORED_TOKEN_ID or input_id == label, "valid token id in input_ids and labels do not match!"

        # check whether target texts match
        target_ids = [list(y) for x, y in groupby(labels, lambda x: x != IGNORED_TOKEN_ID) if x]
        target_texts = map(tokenizer.decode, target_ids)
        assistant_responses = [_["value"] for _ in conv["conversations"] if _["from"] == "gpt"]
        for target_text, assistant_response in zip(target_texts, assistant_responses):
            assert target_text == assistant_response + conversation.role_ends["gpt"], "target text and gpt response do not match!"

    print("check passed!")


if __name__ == "__main__":
    fire.Fire()
