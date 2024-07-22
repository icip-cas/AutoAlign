"""finetune"""
import json
from functools import partial
from dataclasses import dataclass, field
import pathlib
import random

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

from autoalign.train.utils import configure_model

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
    num_workers: str = field(default=8, metadata={"help": "number of workers for tokenization"})

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

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
    
    global local_rank
    local_rank = training_args.local_rank
    rank0_print(f"{model_args = }")
    rank0_print(f"{data_args = }")

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

    dataset = dataset.select(range(100))

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

    random_idx = random.randint(0, len(dataset))
    input_ids = dataset[random_idx]["input_ids"]
    input_text = tokenizer.decode(input_ids)
    rank0_print("-----------Full Text-----------")
    rank0_print(input_text)
    rank0_print("-----------Train on Text-----------")
    labels = dataset[random_idx]["labels"]
    mask = [label != -100 for label in labels]
    train_ids = [idx for idx, m in zip(input_ids, mask) if m]
    train_text = tokenizer.decode(train_ids)
    rank0_print(train_text)

    # get data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    configure_model(data_args.conv_template_name, tokenizer, model)

    # create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Resume training from existing checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
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