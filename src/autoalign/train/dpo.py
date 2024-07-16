from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
import json
from datasets import Dataset

from datasets import load_dataset
from functools import partial

from trl import DPOTrainer, DPOConfig
from autoalign.conversation import Conversation
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast

from autoalign.train.utils import adpative_config

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    conv_template_name: str = field(default="qwen-7b-chat", metadata={"help": "name of conversation template"})


def preprocess(sample, conv_template_name):
    prompt_conversations = Conversation.from_template(conv_template_name)
    chosen_conversations = Conversation.from_template(conv_template_name)
    rejected_conversations = Conversation.from_template(conv_template_name)
    if "system" in sample:
        prompt_conversations.overwrite_system_message = sample["system"]
        chosen_conversations.overwrite_system_message = sample["system"]
        rejected_conversations.overwrite_system_message = sample["system"]

    prompt_conversations.fill_in_messages({"conversations": sample["chosen"][:1]})
    chosen_conversations.fill_in_messages({"conversations": sample["chosen"]})
    rejected_conversations.fill_in_messages({"conversations": sample["rejected"]})

    return dict(
        prompt=prompt_conversations.get_conversation_str(add_generation_prompt=True),
        chosen=chosen_conversations.get_conversation_str(),
        rejected=rejected_conversations.get_conversation_str(),
    )


def run_dpo():
    # parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"{model_args=}")
    print(f"{data_args=}")

    # read data
    with open(data_args.data_path, "r") as f:
        data = json.load(f)

    # get dataset
    dataset = Dataset.from_list(data)

    # load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    model_refer = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    adpative_config(data_args.conv_template_name, tokenizer, model)

    # NB: use eos_token for padding
    tokenizer.pad_token = tokenizer.eos_token
    # set padding_side
    tokenizer.padding_side = "left"
    # specifically set bos_token_id for Qwen2Tokenizer
    if isinstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast)):
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

    # process dataset
    dataset = dataset.map(
        partial(preprocess, conv_template_name=data_args.conv_template_name),
        num_proc=8,
        remove_columns=[col for col in dataset.features if col not in ["prompt", "chosen", "rejected"]],
    )

    # create trainer
    trainer = DPOTrainer(
        model,
        model_refer,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    # start training
    trainer.train()


if __name__ == "__main__":
    run_dpo()
