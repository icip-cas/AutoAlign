import json
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import datasets
import pandas as pd
import torch
import transformers
import trl
from accelerate.state import PartialState
from accelerate.utils import gather_object
from datasets import Dataset
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast
from trl import RewardConfig
from trl.trainer.utils import print_rich_table

from autoalign.conversation import Conversation
from autoalign.train.utils import configure_model


class RewardTrainer(trl.RewardTrainer):
    def visualize_samples(self, num_print_samples: int):
        """
        Visualize the reward model logits prediction

        Args:
            num_print_samples (`int`, defaults to `4`):
                The number of samples to print. Set to `-1` to print all samples.
        """
        if isinstance(self.eval_dataset, dict):
            eval_dataset = next(iter(self.eval_dataset.values()))
        else:
            eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        table = defaultdict(list)
        for _, inputs in enumerate(eval_dataloader):
            _, logits, _ = self.prediction_step(
                self.model, inputs, prediction_loss_only=False
            )
            chosen_text = self.tokenizer.batch_decode(
                inputs["input_ids_chosen"], skip_special_tokens=True
            )
            rejected_text = self.tokenizer.batch_decode(
                inputs["input_ids_rejected"], skip_special_tokens=True
            )
            table["chosen_text"].extend(gather_object(chosen_text))
            table["rejected_text"].extend(gather_object(rejected_text))
            table["logits"].extend(
                gather_object(
                    [
                        [round(inner_item, 4) for inner_item in item]
                        for item in logits.tolist()
                    ]
                )
            )
            if (
                num_print_samples >= 0
                and len(table["chosen_text"]) >= num_print_samples
            ):
                break
        df = pd.DataFrame(table)
        if self.accelerator.process_index == 0:
            print_rich_table(df[:num_print_samples])
            if "wandb" in self.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="pretrained_models/Meta-Llama-3-8B"
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="data/ultra_binary.jsonl",
        metadata={"help": "Path to the training data."},
    )
    eval_path: str = field(
        default="data/eval", metadata={"help": "Path to the training data."}
    )
    conv_template_name: str = field(
        default="llama-3-instruct", metadata={"help": "name of conversation template"}
    )


def prepare_dataset(data_args, training_args, tokenizer):
    def load_single_dataset(name_or_path):
        if os.path.exists(name_or_path):
            if name_or_path.endswith(".json"):
                with open(name_or_path, "r") as f:
                    dataset = json.load(f)
                return Dataset.from_list(dataset)
            elif name_or_path.endswith(".jsonl"):
                return Dataset.from_json(name_or_path)
        else:
            return datasets.load_dataset(name_or_path)["train"]

    train_dataset = load_single_dataset(data_args.data_path)
    eval_dataset = None
    if data_args.eval_path is not None:
        if os.path.exists(data_args.eval_path) and os.path.isdir(data_args.eval_path):
            eval_dataset = {}
            for name in os.listdir(data_args.eval_path):
                eval_dataset[os.path.splitext(name)[0]] = load_single_dataset(
                    os.path.join(data_args.eval_path, name)
                )
        else:
            eval_dataset = load_single_dataset(data_args.eval_path)

    def preprocess(sample):
        prompt_conversations = Conversation.from_template(data_args.conv_template_name)
        prompt_conversations.fill_in_messages({"conversations": sample["chosen"]})
        chosen = prompt_conversations.get_conversation_str()
        prompt_conversations.clear_message()
        prompt_conversations.fill_in_messages({"conversations": sample["rejected"]})
        rejected = prompt_conversations.get_conversation_str()
        tokenized_chosen = tokenizer(chosen)
        tokenized_rejected = tokenizer(rejected)
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    def transform(dataset):
        dataset = dataset.map(
            preprocess,
            num_proc=8,
            remove_columns=[
                col
                for col in dataset.features
                if col
                not in [
                    "input_ids_chosen",
                    "attention_mask_chosen",
                    "input_ids_rejected",
                    "attention_mask_rejected",
                ]
            ],
        )
        dataset = dataset.filter(
            lambda x: (
                len(x["input_ids_chosen"]) <= training_args.max_length
                and len(x["input_ids_rejected"]) <= training_args.max_length
            )
        )
        return dataset

    # process dataset
    with PartialState().local_main_process_first():
        train_dataset = transform(train_dataset)
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {
                    name: transform(eval_dataset[name]) for name in eval_dataset
                }
            else:
                eval_dataset = transform(eval_dataset)

    return train_dataset, eval_dataset


def run_rm():
    # parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, RewardConfig)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(f"{model_args=}")
    print(f"{data_args=}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    # NB: use eos_token for padding
    tokenizer.pad_token = tokenizer.eos_token
    # set padding_side
    tokenizer.padding_side = "left"
    # specifically set bos_token_id for Qwen2Tokenizer
    if isinstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast)):
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    configure_model(data_args.conv_template_name, tokenizer, model)

    train_dataset, eval_dataset = prepare_dataset(
        data_args=data_args,
        training_args=training_args,
        tokenizer=tokenizer,
    )

    # create trainer
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resume training from existing checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)


if __name__ == "__main__":
    run_rm()
