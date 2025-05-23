"""finetune"""
import json
from tqdm.auto import tqdm
from functools import partial
from itertools import groupby, chain
from dataclasses import dataclass, field
import pathlib
import random
from accelerate.state import PartialState
from typing import Dict
from multiprocessing import Pool
import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
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
from autoalign.train.patch import patch_for_block_diag_attn
from autoalign.train.utils import (
    configure_model,
    split_list,
    greedy_knapsack,
    pack_data_points_by_length,
)

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
    num_workers: int = field(
        default=8, metadata={"help": "number of workers for tokenization"}
    )
    lazy_preprocess: bool = False
    eval_num: int = field(
        default=0, metadata={"help": "number of data points for evaluation"}
    )
    # ref: llama-factory
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    packing_strategy: str = field(
        default="sequentially",
        metadata={
            "help": 'The strategy of packing the data (merge short sequences into a long one). Available: "greedy" and "sequentially"'
        },
    )
    # seed: int = field(
    #     default=42, metadata={"help": "random seed"}
    # )


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

    tokenized_conversation["attention_mask"] = [1] * len(
        tokenized_conversation["input_ids"]
    )

    return tokenized_conversation


def packing_data(numbers: list[int], dataset: list):
    packed_input_ids, packed_attention_masks, packed_labels = [], [], []

    for idx, num in enumerate(numbers):
        packed_input_ids += dataset[num]["input_ids"]
        packed_labels += dataset[num]["labels"]
        packed_attention_masks += [idx + 1] * len(
            dataset[num]["input_ids"]
        )  # start from 1, 2 ...

    return {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_masks,
        "labels": packed_labels,
    }


class LazySupervisedDataset(TorchDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer: transformers.PreTrainedTokenizer,
        conv_template_name: str,
        model_max_length: int,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.conv_template_name = conv_template_name
        self.model_max_length = model_max_length

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        ret = tokenize_conversation(
            self.raw_data[i],
            self.conv_template_name,
            self.tokenizer,
            self.model_max_length,
        )

        self.cached_data_dict[i] = ret

        return ret


def run_sft():
    # parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    global local_rank
    local_rank = training_args.local_rank
    rank0_print(f"{model_args = }")
    rank0_print(f"{data_args = }")

    # set random seed for reproducibility
    random.seed(training_args.data_seed)

    # read data
    with open(data_args.data_path, "r") as f:
        data = json.load(f)

    # split data into train and dev datasets
    if data_args.eval_num > 0:
        random.shuffle(data)
        train_data = data[: -data_args.eval_num]
        dev_data = data[-data_args.eval_num :]
    else:
        train_data = data
        dev_data = []
        training_args.eval_strategy = "no"

    rank0_print(f"Train dataset size: {len(train_data)}")
    rank0_print(f"Dev dataset size: {len(dev_data)}")

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    if config.model_type == "gemma2":
        attn_implementation = "eager"
    else:
        attn_implementation = "flash_attention_2"

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # FIXME: currently use bfloat16 regardless of training script
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    # NB: use eos_token for padding
    tokenizer.pad_token = tokenizer.eos_token
    # set padding_side
    tokenizer.padding_side = "right" if config.model_type != "chatglm" else "left"
    # specifically set bos_token_id for Qwen2Tokenizer
    if isinstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast)):
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

    # get dataset
    if data_args.lazy_preprocess:
        train_dataset = LazySupervisedDataset(
            train_data,
            tokenizer=tokenizer,
            conv_template_name=data_args.conv_template_name,
            model_max_length=model_args.model_max_length,
        )

        dev_dataset = LazySupervisedDataset(
            dev_data,
            tokenizer=tokenizer,
            conv_template_name=data_args.conv_template_name,
            model_max_length=model_args.model_max_length,
        )

        rank0_print("Loading data...")

    else:
        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data)
        # tokenize dataset
        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(
                partial(
                    tokenize_conversation,
                    conv_template_name=data_args.conv_template_name,
                    tokenizer=tokenizer,
                    model_max_length=model_args.model_max_length,
                ),
                remove_columns=list(train_dataset.features),
                num_proc=data_args.num_workers,
            )
            dev_dataset = dev_dataset.map(
                partial(
                    tokenize_conversation,
                    conv_template_name=data_args.conv_template_name,
                    tokenizer=tokenizer,
                    model_max_length=model_args.model_max_length,
                ),
                remove_columns=list(dev_dataset.features),
                num_proc=data_args.num_workers,
            )
            if data_args.neat_packing:
                rank0_print("-----------Start Knapsacking-----------")
                lengths = [
                    (idx, len(train_dataset[idx]["input_ids"]))
                    for idx in range(len(train_dataset))
                ]
                lengths_para = split_list(lengths, data_args.num_workers)
                with Pool(data_args.num_workers) as p:
                    if data_args.packing_strategy == "greedy":
                        knapsacks_para = p.starmap_async(
                            greedy_knapsack,
                            tqdm(
                                [
                                    (para, model_args.model_max_length - 1)
                                    for para in lengths_para
                                ]
                            ),
                        )
                    elif data_args.packing_strategy == "sequentially":
                        knapsacks_para = p.starmap_async(
                            pack_data_points_by_length,
                            tqdm(
                                [
                                    (para, model_args.model_max_length - 1)
                                    for para in lengths_para
                                ]
                            ),
                        )
                    else:
                        raise NotImplementedError(
                            'Invalid packing strategy. Available: "greedy" and "sequentially"'
                        )
                    knapsacks = [knap for knap in knapsacks_para.get()]
                    knapsacks = list(chain(*knapsacks))
                    p.close()
                    p.join()
                rank0_print("-----------Start Packing-----------")
                with Pool(data_args.num_workers) as p:
                    packing_train_data = p.starmap_async(
                        packing_data,
                        tqdm([(knapsack, train_dataset) for knapsack in knapsacks]),
                    )
                    packed_train_data = [pack for pack in packing_train_data.get()]
                    p.close()
                    p.join()
                train_dataset = Dataset.from_list(packed_train_data)
                patch_for_block_diag_attn(model_args.model_name_or_path)
                rank0_print("-----------Packing Completed-----------")

    random_idx = random.randint(0, len(train_dataset) - 1)
    input_ids = train_dataset[random_idx]["input_ids"]
    input_text = tokenizer.decode(input_ids)
    rank0_print("-----------Full Text-----------")
    rank0_print(input_text)
    rank0_print("-----------Train on Text-----------")
    labels = train_dataset[random_idx]["labels"]
    target_ids = [list(y) for x, y in groupby(labels, lambda x: x != -100) if x]
    target_texts = list(map(tokenizer.decode, target_ids))
    rank0_print("\n>>>>>>>>>>>>>>>>>\n".join(target_texts))

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
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
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
