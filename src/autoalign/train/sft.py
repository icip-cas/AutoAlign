"""finetune"""
import json
from functools import partial
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
from fastalign.conversation import Conversation

# model related args
@dataclass
class ModelArguments:
    model_name_or_path: str
    model_max_length: int

# data related args
@dataclass
class DataArguments:
    data_path: str
    conv_template_name: str = field(
        metadata={"help": "name of conversation template"}
    )

def tokenize_conversation(
    conv,
    conv_template_name,
    tokenizer: AutoTokenizer,
    model_max_length: int,
):
    """tokenize conversation and prepare labels for loss calculation"""
    # get conversation template
    conversation = Conversation.from_template(conv_template_name)

    # fill in messages from single conv data point
    conversation.fill_in_messages(conv)

    # tokenize conversation
    tokenized_conversation = conversation.get_tokenized_conversation(
        tokenizer=tokenizer,
        model_max_length=model_max_length,
    )

    return tokenized_conversation

def main():
    # parse arguments
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"{model_args=}")
    print(f"{data_args=}")

    # read data
    with open(data_args.data_path, 'r') as f:
        data = json.load(f)

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # FIXME: currently use bfloat16 regardless of training script
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    # NB: use eos_token for padding
    tokenizer.pad_token = tokenizer.eos_token
    # set padding_side
    tokenizer.padding_side  = 'left'

    # get dataset
    dataset = Dataset.from_list(data)

    # tokenize dataset
    dataset = dataset.map(partial(tokenize_conversation, conv_template_name=data_args.conv_template_name, tokenizer=tokenizer, model_max_length=model_args.model_max_length), remove_columns=list(dataset.features), num_proc=8)

    # get data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # start training
    trainer.train()

if __name__=='__main__':
    main()
