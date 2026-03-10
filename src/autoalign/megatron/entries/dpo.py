# Copyright (c) 2024 AutoAlign Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Model-agnostic DPO entry point for Megatron training.

Model type is resolved from ``--model-type`` (explicit) or ``--model-path``
(auto-derived from HF config.json).
"""

import autoalign.megatron  # noqa: F401  # bootstrap MEGATRON_LM_PATH before megatron imports

import os
from functools import partial

import torch
import torch._dynamo
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
)

from autoalign.megatron.patch.training_dpo import dpo
from autoalign.megatron.patch.data.gpt_dataset_dpo import build_train_valid_test_datasets_dpo
from autoalign.megatron.patch.data.online_dataset import build_train_valid_test_datasets_online_dpo
from autoalign.megatron.patch.data.utils import get_batch_on_this_tp_rank_idxmap_dpo
from autoalign.megatron.patch.model.qwen2.model_dpo import GPTModelDPO
from autoalign.megatron.patch.arguments import get_patch_args
from autoalign.megatron.registry import make_model_provider

torch._dynamo.config.suppress_errors = True

# model_type resolved from --model-path / --model-type at runtime
model_provider = make_model_provider(
    model_cls=GPTModelDPO,
    extra_args_fn=lambda args: dict(
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
    ),
)


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank_idxmap_dpo(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    packed_seq_params = None

    return tuple([*batch.values(), packed_seq_params])



def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()
    loss, metrics = output_tensor

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model: GPTModelDPO):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModelDPO): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator)
    timers("batch-generator").stop()


    output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)

    return output_tensor, partial(loss_func, loss_mask)


def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0


def train_valid_test_datasets_provider(train_valid_test_num_samples):
    """Build the train test and validation datasets.
    """
    args = get_args()
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if args.dataset == "json":
        # Online tokenization: read JSON + tokenize on-the-fly
        assert args.model_path is not None, "--model-path is required for --dataset json"
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_online_dpo(
            data_path=args.data_path[0],
            model_path=args.model_path,
            template_name=args.template,
            seq_length=args.seq_length,
            seed=args.seed,
            splits_string=args.split,
            epochs=args.epochs,
            shuffle_all_epochs=args.shuffle_all_epochs or False,
        )
    else:
        # Offline: load pre-tokenized MMap dataset
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_dpo(
            data_prefix=args.data_path,
            data_impl=args.dataset,
            splits_string=args.split,
            seq_length=args.seq_length,
            seed=args.seed,
        )

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    train_valid_test_datasets_provider.is_distributed = True

    dpo(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
