# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Qihoo 360 Corporation and HaoshengZou
# SPDX-License-Identifier: Apache-2.0
# modified from https://github.com/Qihoo360/360-LLaMA-Factory

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler
from transformers import Trainer
from transformers.trainer import _is_peft_model
from typing_extensions import override

import importlib.metadata
import importlib.util
from functools import lru_cache
from typing import TYPE_CHECKING

from packaging import version

IGNORE_INDEX = -100

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def _get_package_version(name: str):
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")

@lru_cache
def is_transformers_version_equal_to_4_46():
    return version.parse("4.46.0") <= _get_package_version("transformers") <= version.parse("4.46.1")
    
@lru_cache
def is_transformers_version_greater_than_4_51():
    return version.parse("4.51.0") <= _get_package_version("transformers")

# Monkey patch functions
def _custom_get_train_sampler(self, train_dataset):
    if self.model.sequence_parallel_group is not None:
        return SequentialSampler(self.train_dataset)
    else:
        # Call the original method
        return self._original_get_train_sampler()

def _custom_training_step(self, model, inputs, *args, **kwargs):
    if not hasattr(self, '_has_dummy_forwarded'):
        self._has_dummy_forwarded = False
    # TODO: sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
    if not self._has_dummy_forwarded and model.sequence_parallel_group is not None:
        model.eval()
        with torch.no_grad():
            _ = model(**inputs)
        model.train()
        self._has_dummy_forwarded = True
    return Trainer._original_training_step(self, model, inputs, *args, **kwargs)

def _custom_compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    r"""
    Fixes the loss value for transformers 4.46.0.
    https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
    """
    if model.sequence_parallel_group is None:  # no sequence parallel, compute as it is
        loss, outputs = self._original_compute_loss(model, inputs, return_outputs=True, **kwargs)
    else:
        # compute loss without shift labels, as we have already shifted labels in data processing when using sequence parallel
        _, outputs = self._original_compute_loss(model, inputs, return_outputs=True, **kwargs)
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="sum")
        logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]
        # Get vocab_size
        unwrapped_model = self.accelerator.unwrap_model(model)
        if _is_peft_model(unwrapped_model):
            vocab_size = unwrapped_model.base_model.model.config.vocab_size
        else:
            vocab_size = unwrapped_model.config.vocab_size
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
        # Enable model parallelism
        labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)

        # weighted reduce within sequence_parallel_group
        sp_group = model.sequence_parallel_group
        loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
        label_num = (labels != loss_fct.ignore_index).sum()
        label_num = dist.nn.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
        loss /= label_num

    # now is single-sequence loss
    # print('###loss###', loss.shape, loss)

        if is_transformers_version_greater_than_4_51():
            loss = loss / self.args.gradient_accumulation_steps

    if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
        # other model should not scale the loss
        if return_outputs:
            return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
        else:
            return loss / self.args.gradient_accumulation_steps

    return loss


def apply_custom_trainer_patches():
    """
    Apply monkey patches to Trainer class
    """
    # Store original methods
    if not hasattr(Trainer, '_original_get_train_sampler'):
        Trainer._original_get_train_sampler = Trainer._get_train_sampler
    if not hasattr(Trainer, '_original_compute_loss'):
        Trainer._original_compute_loss = Trainer.compute_loss
    if not hasattr(Trainer, '_original_training_step'):
        Trainer._original_training_step = Trainer.training_step
    
    # Apply monkey patches
    Trainer._get_train_sampler = _custom_get_train_sampler
    Trainer.compute_loss = _custom_compute_loss
    Trainer.training_step = _custom_training_step

def remove_custom_trainer_patches():
    """
    Remove monkey patches and restore original methods
    """
    if hasattr(Trainer, '_original_get_train_sampler'):
        Trainer._get_train_sampler = Trainer._original_get_train_sampler
        delattr(Trainer, '_original_get_train_sampler')
    
    if hasattr(Trainer, '_original_compute_loss'):
        Trainer.compute_loss = Trainer._original_compute_loss
        delattr(Trainer, '_original_compute_loss')

    if hasattr(Trainer, '_original_training_step'):
        Trainer.training_step = Trainer._original_training_step
        delattr(Trainer, '_original_training_step')

if __name__ == "__main__":
    apply_custom_trainer_patches()

    # remove_custom_trainer_patches()
