"""finetune"""
import json
import os
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
from autoalign.conversation import Conversation
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast
import transformers

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# model related args
@dataclass
class ModelArguments:
    model_name_or_path: str
    model_max_length: int


# data related args
@dataclass
class DataArguments:
    data_path: str
    conv_template_name: str = field(metadata={"help": "name of conversation template"})

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def adpative_config(conv_template_name, tokenizer, model):
    """ specify eos token and bos token for model and tokenizer based on conversation template """
    conversation = Conversation.from_template(conv_template_name)
    eos_token = conversation.role_ends["gpt"].strip()
    eos_token_id = tokenizer(eos_token).input_ids[-1]
    # print(f"{tokenizer(eos_token)=} {eos_token=} {eos_token_id=} {tokenizer.decode([eos_token_id])=}")

    assert eos_token == tokenizer.decode([eos_token_id]), "eos token is not a valid token"
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token
    
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    
    return None

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
    # print(conversation.get_conversation_str())
    # print(tokenized_conversation)

    return tokenized_conversation

def run_sft():
    # parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    rank0_print(f"{model_args=}")
    rank0_print(f"{data_args=}")

    # read data
    with open(data_args.data_path, "r") as f:
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
    tokenizer.padding_side = "right"
    # specifically set bos_token_id for Qwen2Tokenizer
    if isinstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast)):
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

    # get dataset
    dataset = Dataset.from_list(data)

    # tokenize dataset
    dataset = dataset.map(
        partial(
            tokenize_conversation,
            conv_template_name=data_args.conv_template_name,
            tokenizer=tokenizer,
            model_max_length=model_args.model_max_length,
        ),
        remove_columns=list(dataset.features),
        num_proc=data_args.num_workers,
    )

    rank0_print(dataset[0])

    # get data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    adpative_config(data_args.conv_template_name, tokenizer, model)

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

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)
    
if __name__ == "__main__":
    run_sft()