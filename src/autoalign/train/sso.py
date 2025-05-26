from dataclasses import dataclass, field
from typing import Optional
import pathlib
from accelerate.state import PartialState

import torch
import torch.nn as nn
import torch.amp as amp
import torch.nn.functional as F
import numpy as np
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils import is_peft_available, is_torch_xpu_available
from contextlib import contextmanager, nullcontext

import json
from datasets import Dataset, IterableDataset

from functools import partial

from trl import DPOTrainer, DPOConfig
from autoalign.conversation import Conversation
from transformers import Qwen2Tokenizer, Qwen2TokenizerFast

from autoalign.train.utils import configure_model

from typing import Any, Callable, Literal, Optional, Union, Dict, Sequence

from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    is_comet_available,
    is_wandb_available,
)

from transformers.data.data_collator import DataCollatorMixin

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    # Determine the maximum shape for each dimension
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        # Determine the slice for the sequence dimension
        if padding_side == "left":
            seq_slice = slice(output_shape[0] - t.shape[0], output_shape[0])
        elif padding_side == "right":
            seq_slice = slice(0, t.shape[0])
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )

def flush_left(mask: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    # Create copy of mask and tensors
    mask = mask.clone()
    tensors = [t.clone() for t in tensors]

    # Shift non-zero values to the left
    for i in range(mask.size(0)):
        first_one_idx = torch.nonzero(mask[i])[0].item()
        mask[i] = torch.roll(mask[i], shifts=-first_one_idx)
        for tensor in tensors:
            tensor[i] = torch.roll(tensor[i], shifts=-first_one_idx)

    # Get the first column idx that is all zeros and remove every column after that
    empty_cols = torch.sum(mask, dim=0) == 0
    first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else mask.size(1)
    mask = mask[:, :first_empty_col]
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor[:, :first_empty_col]

    if not tensors:
        return mask
    else:
        return mask, *tensors

def selective_log_softmax(logits, index):

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def process_inputs_and_get_logits(self, attention_mask, input_ids, loss_mask):
    model_kwargs = {}
    if self.aux_loss_enabled:
        model_kwargs["output_router_logits"] = True

    if self.use_logits_to_keep:
        # 找到 loss_mask 第一个为 True 的位置
        first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
        logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 包含第一个label
        model_kwargs["logits_to_keep"] = logits_to_keep

    position_ids = 0
    if self.padding_free:
        # 压缩输入（去除 padding），重新生成 position_ids
        input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
        loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
        position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
        model_kwargs["position_ids"] = position_ids
    else:
        model_kwargs["attention_mask"] = attention_mask

    # 调用模型前向推理
    outputs = self.model(input_ids, **model_kwargs)
    logits = outputs.logits

    # 将 input_ids 和 loss_mask 向左滚动一位，得到对应的 labels 和 loss_mask
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

    if self.use_logits_to_keep:
        # 根据 logits_to_keep 切片 labels 和 loss_mask，只保留需要的部分
        labels = labels[:, -logits_to_keep:]
        loss_mask = loss_mask[:, -logits_to_keep:]

    return outputs, logits, labels, loss_mask, position_ids


@dataclass
class SSODataCollatorWithPadding(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        prompt_input_ids = [torch.tensor(example["prompt_input_ids"]) for example in examples]
        prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in prompt_input_ids]
        chosen_input_ids = [torch.tensor(example["chosen_input_ids"]) for example in examples]
        chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in chosen_input_ids]
        rejected_input_ids = [torch.tensor(example["rejected_input_ids"]) for example in examples]
        rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in rejected_input_ids]

        p_prompt_input_ids = [torch.tensor(example["p_prompt_input_ids"]) for example in examples]
        p_prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in p_prompt_input_ids]
        p_chosen_input_ids = [torch.tensor(example["p_chosen_input_ids"]) for example in examples]
        p_chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in p_chosen_input_ids]
        p_rejected_input_ids = [torch.tensor(example["p_rejected_input_ids"]) for example in examples]
        p_rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in p_rejected_input_ids]

        n_prompt_input_ids = [torch.tensor(example["n_prompt_input_ids"]) for example in examples]
        n_prompt_attention_mask = [torch.ones_like(input_ids) for input_ids in n_prompt_input_ids]
        n_chosen_input_ids = [torch.tensor(example["n_chosen_input_ids"]) for example in examples]
        n_chosen_attention_mask = [torch.ones_like(input_ids) for input_ids in n_chosen_input_ids]
        n_rejected_input_ids = [torch.tensor(example["n_rejected_input_ids"]) for example in examples]
        n_rejected_attention_mask = [torch.ones_like(input_ids) for input_ids in n_rejected_input_ids]

        # Pad
        output = {}
        output["prompt_input_ids"] = pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["prompt_attention_mask"] = pad(prompt_attention_mask, padding_value=0, padding_side="left")
        output["chosen_input_ids"] = pad(chosen_input_ids, padding_value=self.pad_token_id)
        output["chosen_attention_mask"] = pad(chosen_attention_mask, padding_value=0)
        output["rejected_input_ids"] = pad(rejected_input_ids, padding_value=self.pad_token_id)
        output["rejected_attention_mask"] = pad(rejected_attention_mask, padding_value=0)
        output["p_prompt_input_ids"] = pad(p_prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["p_prompt_attention_mask"] = pad(p_prompt_attention_mask, padding_value=0, padding_side="left")
        output["p_chosen_input_ids"] = pad(p_chosen_input_ids, padding_value=self.pad_token_id)
        output["p_chosen_attention_mask"] = pad(p_chosen_attention_mask, padding_value=0)
        output["p_rejected_input_ids"] = pad(p_rejected_input_ids, padding_value=self.pad_token_id)
        output["p_rejected_attention_mask"] = pad(p_rejected_attention_mask, padding_value=0)
        output["n_prompt_input_ids"] = pad(n_prompt_input_ids, padding_value=self.pad_token_id, padding_side="left")
        output["n_prompt_attention_mask"] = pad(n_prompt_attention_mask, padding_value=0, padding_side="left")
        output["n_chosen_input_ids"] = pad(n_chosen_input_ids, padding_value=self.pad_token_id)
        output["n_chosen_attention_mask"] = pad(n_chosen_attention_mask, padding_value=0)
        output["n_rejected_input_ids"] = pad(n_rejected_input_ids, padding_value=self.pad_token_id)
        output["n_rejected_attention_mask"] = pad(n_rejected_attention_mask, padding_value=0)

        return output

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen2/Qwen2-7B-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template_name: str = field(
        default="chatml", metadata={"help": "name of conversation template"}
    )

class SSOTrainer(DPOTrainer):
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: DPOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Build the kwargs for the `map` function
        map_kwargs = {"writer_batch_size": 10}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().local_main_process_first():
            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row, # 需要修改这里的self.tokenize_row函数
                remove_columns=["prompt", "chosen", "rejected", "p_prompt", "p_chosen", "p_rejected", "n_prompt", "n_chosen", "n_rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset

    @staticmethod
    def tokenize_row(features, processing_class, max_prompt_length, max_completion_length, add_special_tokens):
        tokenizer = processing_class  # the processing class is a tokenizer

        prompt_input_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
        chosen_input_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
        rejected_input_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

        p_prompt_input_ids = tokenizer(features["p_prompt"], add_special_tokens=False)["input_ids"]
        p_chosen_input_ids = tokenizer(features["p_chosen"], add_special_tokens=False)["input_ids"]
        p_rejected_input_ids = tokenizer(features["p_rejected"], add_special_tokens=False)["input_ids"]

        n_prompt_input_ids = tokenizer(features["n_prompt"], add_special_tokens=False)["input_ids"]
        n_chosen_input_ids = tokenizer(features["n_chosen"], add_special_tokens=False)["input_ids"]
        n_rejected_input_ids = tokenizer(features["n_rejected"], add_special_tokens=False)["input_ids"]

        # Add special tokens (typically for encoder-decoder models)
        if add_special_tokens:
            if tokenizer.bos_token_id is not None:
                prompt_input_ids = [tokenizer.bos_token_id] + prompt_input_ids
                p_prompt_input_ids = [tokenizer.bos_token_id] + p_prompt_input_ids
                n_prompt_input_ids = [tokenizer.bos_token_id] + n_prompt_input_ids
            if tokenizer.eos_token_id is not None:
                prompt_input_ids = prompt_input_ids + [tokenizer.eos_token_id]
                p_prompt_input_ids = p_prompt_input_ids + [tokenizer.eos_token_id]
                n_prompt_input_ids = n_prompt_input_ids + [tokenizer.eos_token_id]
        
        chosen_input_ids = chosen_input_ids + [tokenizer.eos_token_id]
        rejected_input_ids = rejected_input_ids + [tokenizer.eos_token_id]

        p_chosen_input_ids = p_chosen_input_ids + [tokenizer.eos_token_id]
        p_rejected_input_ids = p_rejected_input_ids + [tokenizer.eos_token_id]

        n_chosen_input_ids = n_chosen_input_ids + [tokenizer.eos_token_id]
        n_rejected_input_ids = n_rejected_input_ids + [tokenizer.eos_token_id]

        # Truncate prompt and completion sequences
        if max_prompt_length is not None:
            prompt_input_ids = prompt_input_ids[-max_prompt_length:]
            p_prompt_input_ids = p_prompt_input_ids[-max_prompt_length:]
            n_prompt_input_ids = n_prompt_input_ids[-max_prompt_length:]
        if max_completion_length is not None:
            chosen_input_ids = chosen_input_ids[:max_completion_length]
            rejected_input_ids = rejected_input_ids[:max_completion_length]
            p_chosen_input_ids = p_chosen_input_ids[:max_completion_length]
            p_rejected_input_ids = p_rejected_input_ids[:max_completion_length]
            n_chosen_input_ids = n_chosen_input_ids[:max_completion_length]
            n_rejected_input_ids = n_rejected_input_ids[:max_completion_length]

        return {
            "prompt_input_ids": prompt_input_ids,
            "chosen_input_ids": chosen_input_ids,
            "rejected_input_ids": rejected_input_ids,
            "p_prompt_input_ids": p_prompt_input_ids,
            "p_chosen_input_ids": p_chosen_input_ids,
            "p_rejected_input_ids": p_rejected_input_ids,
            "n_prompt_input_ids": n_prompt_input_ids,
            "n_chosen_input_ids": n_chosen_input_ids,
            "n_rejected_input_ids": n_rejected_input_ids,
        }
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by `DataCollatorForPreference`, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
                "p_prompt_input_ids",
                "p_chosen_input_ids",
                "p_rejected_input_ids",
                "n_prompt_input_ids",
                "n_chosen_input_ids",
                "n_rejected_input_ids",
                "ref_chosen_logps",
                "ref_rejected_logps",
            ]

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, Union[list, torch.LongTensor]], padding_value: int
    ) -> dict[str, torch.LongTensor]:
        output = {}

        # For the prompt, the input_ids are the same for both the chosen and rejected responses
        output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
        output["prompt_attention_mask"] = torch.cat(
            [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
        )

        output["p_prompt_input_ids"] = torch.cat([batch["p_prompt_input_ids"], batch["p_prompt_input_ids"]], dim=0)
        output["p_prompt_attention_mask"] = torch.cat(
            [batch["p_prompt_attention_mask"], batch["p_prompt_attention_mask"]], dim=0
        )
        
        output["n_prompt_input_ids"] = torch.cat([batch["n_prompt_input_ids"], batch["n_prompt_input_ids"]], dim=0)
        output["n_prompt_attention_mask"] = torch.cat(
            [batch["n_prompt_attention_mask"], batch["n_prompt_attention_mask"]], dim=0
        )

        # Concatenate the chosen and rejected completions
        max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
        output["completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["chosen_input_ids"], max_completion_length, pad_value=padding_value),
                pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
            ),
        )
        output["completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        max_completion_length = max(batch["p_chosen_input_ids"].shape[1], batch["p_rejected_input_ids"].shape[1])
        output["p_completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["p_chosen_input_ids"], max_completion_length, pad_value=0),
                pad_to_length(batch["p_rejected_input_ids"], max_completion_length, pad_value=0),
            ),
        )
        output["p_completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["p_chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["p_rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        max_completion_length = max(batch["n_chosen_input_ids"].shape[1], batch["n_rejected_input_ids"].shape[1])
        output["n_completion_input_ids"] = torch.cat(
            (
                pad_to_length(batch["n_chosen_input_ids"], max_completion_length, pad_value=0),
                pad_to_length(batch["n_rejected_input_ids"], max_completion_length, pad_value=0),
            ),
        )
        output["n_completion_attention_mask"] = torch.cat(
            (
                pad_to_length(batch["n_chosen_attention_mask"], max_completion_length, pad_value=0),
                pad_to_length(batch["n_rejected_attention_mask"], max_completion_length, pad_value=0),
            ),
        )

        return output

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        num_examples = batch["prompt_input_ids"].shape[0] # 获得batch size

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        p_prompt_input_ids = concatenated_batch["p_prompt_input_ids"]
        p_prompt_attention_mask = concatenated_batch["p_prompt_attention_mask"]
        n_prompt_input_ids = concatenated_batch["n_prompt_input_ids"]
        n_prompt_attention_mask = concatenated_batch["n_prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        p_completion_input_ids = concatenated_batch["p_completion_input_ids"]
        p_completion_attention_mask = concatenated_batch["p_completion_attention_mask"]
        n_completion_input_ids = concatenated_batch["n_completion_input_ids"]
        n_completion_attention_mask = concatenated_batch["n_completion_attention_mask"]

        if self.is_encoder_decoder: # 没有进行修改
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            model_kwargs = {}
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)

            p_input_ids = torch.cat((p_prompt_input_ids, p_completion_input_ids), dim=1)
            p_attention_mask = torch.cat((p_prompt_attention_mask, p_completion_attention_mask), dim=1)

            n_input_ids = torch.cat((n_prompt_input_ids, n_completion_input_ids), dim=1)
            n_attention_mask = torch.cat((n_prompt_attention_mask, n_completion_attention_mask), dim=1)

            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )
            p_loss_mask = torch.cat(
                (torch.zeros_like(p_prompt_attention_mask), p_completion_attention_mask),
                dim=1,
            )
            n_loss_mask = torch.cat(
                (torch.zeros_like(n_prompt_attention_mask), n_completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            p_attention_mask, p_input_ids, p_loss_mask = flush_left(p_attention_mask, p_input_ids, p_loss_mask)
            n_attention_mask, n_input_ids, n_loss_mask = flush_left(n_attention_mask, n_input_ids, n_loss_mask)

            # Truncate right
            if self.max_length is not None:
                if self.truncation_mode == "keep_end":
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    p_input_ids = p_input_ids[:, -self.max_length :]
                    p_attention_mask = p_attention_mask[:, -self.max_length :]
                    p_loss_mask = p_loss_mask[:, -self.max_length :]
                    n_input_ids = n_input_ids[:, -self.max_length :]
                    n_attention_mask = n_attention_mask[:, -self.max_length :]
                    n_loss_mask = n_loss_mask[:, -self.max_length :]
                elif self.truncation_mode == "keep_start":
                    input_ids = input_ids[:, : self.max_length]
                    attention_mask = attention_mask[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                    p_input_ids = p_input_ids[:, : self.max_length]
                    p_attention_mask = p_attention_mask[:, : self.max_length]
                    p_loss_mask = p_loss_mask[:, : self.max_length]
                    n_input_ids = n_input_ids[:, : self.max_length]
                    n_attention_mask = n_attention_mask[:, : self.max_length]
                    n_loss_mask = n_loss_mask[:, : self.max_length]
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            
            outputs, logits, labels, loss_mask, position_ids = process_inputs_and_get_logits(self, attention_mask, input_ids, loss_mask)
            p_outputs, p_logits, p_labels, p_loss_mask, p_position_ids = process_inputs_and_get_logits(self, p_attention_mask, p_input_ids, p_loss_mask)
            n_outputs, n_logits, n_labels, n_loss_mask, n_position_ids = process_inputs_and_get_logits(self, n_attention_mask, n_input_ids, n_loss_mask)

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=outputs.logits.device, dtype=outputs.logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_

        all_logps = per_token_logps.sum(-1)
        mean_logps = per_token_logps.mean(-1)

        # Compute the log probabilities of the p_labels
        p_labels[~p_loss_mask] = 0
        p_per_token_logps = selective_log_softmax(p_logits, p_labels)
        p_per_token_logps[~p_loss_mask] = 0
        p_per_token_logps = torch.roll(p_per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            p_batch_size, p_seq_len = p_attention_mask.shape
            p_per_token_logps_ = torch.zeros(
                p_batch_size, p_seq_len, device=p_outputs.logits.device, dtype=p_outputs.logits.dtype
            )
            p_per_token_logps_[p_attention_mask.bool()] = p_per_token_logps
            p_per_token_logps = p_per_token_logps_

        p_all_logps = p_per_token_logps.sum(-1)
        p_mean_logps = p_per_token_logps.mean(-1)

        # Compute the log probabilities of the n_labels
        n_labels[~n_loss_mask] = 0
        n_per_token_logps = selective_log_softmax(n_logits, n_labels)
        n_per_token_logps[~n_loss_mask] = 0
        n_per_token_logps = torch.roll(n_per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            n_batch_size, n_seq_len = n_attention_mask.shape
            n_per_token_logps_ = torch.zeros(
                n_batch_size, n_seq_len, device=n_outputs.logits.device, dtype=n_outputs.logits.dtype
            )
            n_per_token_logps_[n_attention_mask.bool()] = n_per_token_logps
            n_per_token_logps = n_per_token_logps_

        n_all_logps = n_per_token_logps.sum(-1)
        n_mean_logps = n_per_token_logps.mean(-1)

        output = {}

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["p_chosen_logps"] = p_all_logps[:num_examples]
        output["p_rejected_logps"] = p_all_logps[num_examples:]
        output["n_chosen_logps"] = n_all_logps[:num_examples]
        output["n_rejected_logps"] = n_all_logps[num_examples:]
        output["p_mean_chosen_logps"] = p_mean_logps[:num_examples]
        output["n_mean_chosen_logps"] = n_mean_logps[:num_examples]

        # # Compute the mean logits
        # if self.padding_free:
        #     # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
        #     # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
        #     # and the second half to the rejected tokens.
        #     # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
        #     split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
        #     mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
        #     mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        # else:
        #     mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
        #     mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()
        
        # # Compute the mean p_logits
        # if self.padding_free:
        #     split_idx = (p_position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
        #     p_mean_chosen_logits = p_logits[0, :split_idx][p_loss_mask[0, :split_idx]].mean()
        #     p_mean_rejected_logits = p_logits[0, split_idx:][p_loss_mask[0, split_idx:]].mean()
        # else:
        #     p_mean_chosen_logits = p_logits[:num_examples][p_loss_mask[:num_examples]].mean()
        #     p_mean_rejected_logits = p_logits[num_examples:][p_loss_mask[num_examples:]].mean()

        # # Compute the mean n_logits
        # if self.padding_free:
        #     split_idx = (n_position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
        #     n_mean_chosen_logits = n_logits[0, :split_idx][n_loss_mask[0, :split_idx]].mean()
        #     n_mean_rejected_logits = n_logits[0, split_idx:][n_loss_mask[0, split_idx:]].mean()
        # else:
        #     n_mean_chosen_logits = n_logits[:num_examples][n_loss_mask[:num_examples]].mean()
        #     n_mean_rejected_logits = n_logits[num_examples:][n_loss_mask[num_examples:]].mean()

        # output["mean_chosen_logits"] = mean_chosen_logits
        # output["mean_rejected_logits"] = mean_rejected_logits
        # output["p_mean_chosen_logits"] = p_mean_chosen_logits
        # output["p_mean_rejected_logits"] = p_mean_rejected_logits
        # output["n_mean_chosen_logits"] = n_mean_chosen_logits
        # output["n_mean_rejected_logits"] = n_mean_rejected_logits

        # if self.aux_loss_enabled:
        #     output["aux_loss"] = outputs.aux_loss
        #     output["p_aux_loss"] = p_outputs.aux_loss
        #     output["n_aux_loss"] = n_outputs.aux_loss

        return output

    def compute_ref_log_probs(self, batch: dict[str, torch.LongTensor]) -> dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        device_type = "xpu" if is_torch_xpu_available() else "cuda"
        compte_ref_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        with torch.no_grad(), compte_ref_context_manager:
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_model_output = self.concatenated_forward(self.model, batch)
            else:
                ref_model_output = self.concatenated_forward(self.ref_model, batch)
        return ref_model_output["chosen_logps"], ref_model_output["rejected_logps"], ref_model_output["p_chosen_logps"], ref_model_output["p_rejected_logps"], ref_model_output["n_chosen_logps"], ref_model_output["n_rejected_logps"]

    def get_batch_loss_metrics(
        self,
        model,
        batch: dict[str, Union[list, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        model_output = self.concatenated_forward(model, batch)

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps, p_ref_chosen_logps, p_ref_rejected_logps, n_ref_chosen_logps, n_ref_rejected_logps = self.compute_ref_log_probs(batch)

        p_losses, p_chosen_rewards, p_rejected_rewards = self.dpo_loss(
            model_output["p_chosen_logps"],
            model_output["p_rejected_logps"],
            p_ref_chosen_logps,
            p_ref_rejected_logps,
        )
        n_losses, n_chosen_rewards, n_rejected_rewards = self.dpo_loss(
            model_output["n_chosen_logps"],
            model_output["n_rejected_logps"],
            n_ref_chosen_logps,
            n_ref_rejected_logps,
        )
        gamma = 0.1
        losses = p_losses + n_losses - gamma * (model_output["p_mean_chosen_logps"] + model_output["n_mean_chosen_logps"])
        return losses.mean(), {}

# 这里应该进行修改
def preprocess(sample, conv_template_name):
    # raw
    prompt_conversations = Conversation.from_template(conv_template_name)

    if "system" in sample and sample["system"] is not None:
        prompt_conversations.system_message = sample["system"]

    prompt_conversations.fill_in_messages({"conversations": sample["chosen"][:-1]})
    
    assert sample["chosen"][:-1] == sample["rejected"][:-1]

    # p_prompt_conversations
    p_prompt_conversations = Conversation.from_template(conv_template_name)

    if "p_system" in sample and sample["p_system"] is not None:
        p_prompt_conversations.system_message = sample["p_system"]

    p_prompt_conversations.fill_in_messages({"conversations": sample["chosen"][:-1]})

    # n_prompt_conversations
    n_prompt_conversations = Conversation.from_template(conv_template_name)

    if "n_system" in sample and sample["n_system"] is not None:
        n_prompt_conversations.system_message = sample["n_system"]

    n_prompt_conversations.fill_in_messages({"conversations": sample["chosen"][:-1]})

    return dict(
        prompt=prompt_conversations.get_conversation_str(add_generation_prompt=True),
        chosen=sample["chosen"][-1]["value"],
        rejected=sample["rejected"][-1]["value"],
        p_prompt=p_prompt_conversations.get_conversation_str(add_generation_prompt=True),
        p_chosen=sample["chosen"][-1]["value"],
        p_rejected=sample["rejected"][-1]["value"],
        n_prompt=n_prompt_conversations.get_conversation_str(add_generation_prompt=True),
        n_chosen=sample["raw"][-1]["value"],
        n_rejected=sample["chosen"][-1]["value"]
    )# 这里将n_prompt对应的chosen选为raw 对应的rejected选为chosen

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

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
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    model_refer = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    configure_model(data_args.conv_template_name, tokenizer, model)

    # NB: use eos_token for padding
    tokenizer.pad_token = tokenizer.eos_token
    # set padding_side
    tokenizer.padding_side = "left"
    # specifically set bos_token_id for Qwen2Tokenizer
    if isinstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast)):
        tokenizer.bos_token = "<|im_start|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
    
    # padding value
    if training_args.padding_value is not None:
        padding_value = training_args.padding_value
    else:
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            padding_value = tokenizer.pad_token_id
        elif hasattr(tokenizer, "tokenizer") and tokenizer.tokenizer.pad_token_id is not None:
            padding_value = tokenizer.tokenizer.pad_token_id
        else:
            raise ValueError(
                "`padding_value` is not specified in `DPOConfig`, and `pad_token_id` is missing in the "
                "`processing_class`. Please either set the `padding_value` argument in `DPOConfig`, or set "
                "`tokenizer.pad_token` (e.g., `tokenizer.pad_token = tokenizer.eos_token`) before instantiating "
                "the trainer."
            )
        
    # create data collator
    data_collator = SSODataCollatorWithPadding(pad_token_id=padding_value)

    # process dataset
    with PartialState().local_main_process_first(): # 优先当前主进程
        dataset = dataset.map(
            partial(preprocess, conv_template_name=data_args.conv_template_name),
            num_proc=8,
            remove_columns=[
                col
                for col in dataset.features
                if col not in ["prompt", "chosen", "rejected", "p_prompt", "p_chosen", "p_rejected", "n_prompt", "n_chosen", "n_rejected"]
            ],
        )

    # create trainer
    trainer = SSOTrainer(
        model,
        model_refer,
        data_collator=data_collator,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )

    # start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resume training from existing checkpoint...")
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
    run_dpo()
