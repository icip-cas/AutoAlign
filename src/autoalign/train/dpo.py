from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.model.model_adapter import get_conversation_template
## TODO: remove dependency on fastchat

from datasets import load_dataset
from functools import partial

from autoalign.train.trainer.dpo_trainer import DPOTrainerForQwen

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_split: str = field(
        default=None, metadata={"help": "Chosen split of the training data."}
    )
    
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    
    cache_path: str = field(
        default=None, metadata={"help": "Path to cache the training data."}
    )
    
    num_proc: int = field(
        default=32
    )
    
    template: str = field(default="vicuna-1.1")
    
    json_path: str = field(
        default=None, metadata={"help": "Path to the json file containing the training data."}
    )

    remove_sys_msg: bool = field(
        default=False
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    beta: float = field(default = 0.1, metadata = {
        "help": "Control the deviation from the reference model."
    })
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    min_lr: float = field(
        default=None
    )
    mask_user: bool = field(
        default=True
    )
    save_global_steps: bool = field(
        default=True
    )
    loss_type: str = field(
        default="sigmoid"
    )

local_rank = None

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def preprocess(
    sample,
    template="vicuna-1.1",
    remove_sys_msg=False
) -> Dict:

    conv = get_conversation_template(template)
    if "system" in sample:
        system_message = sample["system"]
    else:
        system_message = ""

    if remove_sys_msg:
        system_message = ""

    roles = {"user": conv.roles[0], "assistant": conv.roles[1]}

    conv.system_message = system_message

    assert len(sample["chosen"]) == len(sample["rejected"]) == 2

    chosen_sources, rejected_sources = sample["chosen"], sample["rejected"]

    conv.append_message(conv.roles[0], sample["prompt"])
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()

    conv.update_last_message(chosen_sources[1]["content"])

    chosen_conversations = conv.get_prompt()

    conv.update_last_message(rejected_sources[1]["content"])

    rejected_conversations = conv.get_prompt()

    if False:
        print("==============")
        print("prompt:", prompt)
        print("chosen_conversations:", chosen_conversations)
        print("rejected_conversations:", rejected_conversations)
        print("==============")
    
    return dict(
        prompt=prompt,
        chosen=chosen_conversations,
        rejected=rejected_conversations,
    )

def make_dpo_dataset(
    data_args: DataArguments,
    sanity_check: bool=False
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    """
    
    data_split: str = data_args.data_split
    num_proc: int = data_args.num_proc
    template: str =  data_args.template
    
    json_path: str = data_args.json_path
    remove_sys_msg: bool = data_args.remove_sys_msg
    
    dataset = load_dataset(
        "json",
        data_files=json_path,
        split=data_split
    )
        
    original_columns = dataset.column_names
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    preprocess_with_template = partial(preprocess, template=template, remove_sys_msg=remove_sys_msg)
    
    return dataset.map(
        preprocess_with_template,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_eval = False
    local_rank = training_args.local_rank
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False
    
    model_refer = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    train_dataset = make_dpo_dataset(data_args=data_args)
    
    trainer = DPOTrainerForQwen(
        model, 
        model_refer, 
        tokenizer=tokenizer, 
        beta=training_args.beta, 
        args=training_args, 
        train_dataset=train_dataset,
        max_prompt_length=512, 
        max_length=training_args.model_max_length,
        loss_type=training_args.loss_type
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)

if __name__ == "__main__":
    train()