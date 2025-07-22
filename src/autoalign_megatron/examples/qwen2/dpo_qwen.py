# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

# Additional modifications and contributions by AutoAlign Team:
# - Added support for multi-turn dialogue training with SFT (Supervised Fine-Tuning).
# - Integrated DPO (Direct Preference Optimization) for alignment training.

# Copyright (c) 2024 AutoAlign Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DPO"""

import os
from functools import partial
from typing import Union

import torch
import torch._dynamo
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
)
from megatron.core.packed_seq_params import PackedSeqParams

from megatron_patch.model.qwen2.layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron_patch.model.qwen2.transformer_config import Qwen2TransformerConfig
from megatron_patch.tokenizer import build_tokenizer

from autoalign_megatron.patch.training_dpo import dpo
from autoalign_megatron.patch.data.gpt_dataset_dpo import build_train_valid_test_datasets_dpo
from autoalign_megatron.patch.data.utils import get_batch_on_this_tp_rank_idxmap_dpo
from autoalign_megatron.patch.model.qwen2.model_dpo import GPTModelDPO
from autoalign_megatron.patch.arguments import get_patch_args

torch._dynamo.config.suppress_errors = True


def model_provider(
    pre_process=True, post_process=True
) -> Union[GPTModelDPO]:

    args = get_args()
    build_tokenizer(args)
    print_rank_0("building qwen2 model ...")

    config = core_transformer_config_from_args(args, Qwen2TransformerConfig)
    use_te = args.transformer_impl == "transformer_engine"

    if use_te:
        print_rank_0("building qwen2 model in TE...")
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
    else:
        print_rank_0("building qwen2 model in Mcore...")
        transformer_layer_spec = get_gpt_layer_local_spec(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )

    model = GPTModelDPO(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
        loss_type=args.loss_type,
    )
        
    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on

    batch = get_batch_on_this_tp_rank_idxmap_dpo(data_iterator)
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
        model (GPTModel_DPO): The GPT Model
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
