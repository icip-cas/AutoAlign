# Copyright (c) 2024 AutoAlign Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Model-agnostic SFT entry point for Megatron training.

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

from autoalign.megatron.patch.data.gpt_dataset_sft_conv import build_train_valid_test_datasets_sft_conv
from autoalign.megatron.patch.data.online_dataset import build_train_valid_test_datasets_online_sft
from autoalign.megatron.patch.data.utils import get_batch_on_this_tp_rank_idxmap_sft_conv
from megatron.training.training import pretrain as sft
from autoalign.megatron.patch.arguments import get_patch_args
from autoalign.megatron.registry import make_model_provider
import autoalign.megatron.patch.core.context_parallel  # noqa: F401  # CP NCCL warmup

# Patch 1: strict=False — tolerate missing/unexpected keys across impl switches
import megatron.training.checkpointing as _ckpt
import megatron.training.training as _training
_orig_load_checkpoint = _ckpt.load_checkpoint

def _patched_load_checkpoint(*args, strict=True, **kwargs):
    return _orig_load_checkpoint(*args, strict=False, **kwargs)

_ckpt.load_checkpoint = _patched_load_checkpoint
_training.load_checkpoint = _patched_load_checkpoint

# Patch 3: SwanLab / WandB logging — inject into Megatron's training_log.
# Megatron's pretrain is used directly for SFT (not our training_sft.py), so
# we monkey-patch here. SwanLab is lazily initialized on the first training_log
# call (after torch.distributed is ready).
_swanlab_run = None
_swanlab_initialized = False

def _maybe_init_swanlab():
    global _swanlab_run, _swanlab_initialized
    if _swanlab_initialized:
        return _swanlab_run
    _swanlab_initialized = True
    try:
        import torch
        args = get_args()
        if 'swanlab' not in getattr(args, 'report_to', []):
            return None
        if torch.distributed.get_rank() != args.world_size - 1:
            return None
        import swanlab
        _swanlab_run = swanlab.init(config=vars(args))
        print_rank_0(f'SwanLab initialized: {_swanlab_run}')
    except Exception as e:
        print_rank_0(f'WARNING: SwanLab init failed ({e}), disabling SwanLab logging')
    return _swanlab_run

_orig_training_log = _training.training_log

def _patched_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate,
                           iteration, loss_scale, report_memory_flag, skipped_iter,
                           grad_norm, params_norm, num_zeros_in_grad):
    _orig_training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate,
                       iteration, loss_scale, report_memory_flag, skipped_iter,
                       grad_norm, params_norm, num_zeros_in_grad)
    swanlab_writer = _maybe_init_swanlab()
    if not swanlab_writer:
        return
    args = get_args()
    if iteration % args.log_interval != 0:
        return
    try:
        if torch.distributed.get_rank() != args.world_size - 1:
            return
    except Exception:
        return
    metrics = {'learning-rate': learning_rate, 'loss-scale': loss_scale}
    if grad_norm is not None:
        metrics['grad-norm'] = grad_norm
    if params_norm is not None:
        metrics['params-norm'] = params_norm
    for key in loss_dict:
        metrics[key] = loss_dict[key]
    swanlab_writer.log(metrics, step=iteration)

_training.training_log = _patched_training_log

# Patch 2: skip empty TE _extra_state (zero bytes → EOFError in pickle.loads).
# TE _extra_state is only populated after the first forward pass (quantization
# calibration), so freshly converted or newly created checkpoints always have
# empty _extra_state tensors.  This is expected; just skip them on load.
try:
    import transformer_engine.pytorch.module.base as _te_base
    _te_orig_load = _te_base.TransformerEngineBaseModule._load_from_state_dict

    def _te_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "_extra_state"
        if key in state_dict and state_dict[key].nbytes == 0:
            state_dict = {k: v for k, v in state_dict.items() if k != key}
        return _te_orig_load(self, state_dict, prefix, *args, **kwargs)

    _te_base.TransformerEngineBaseModule._load_from_state_dict = _te_load_from_state_dict
except ImportError:
    pass  # TE not installed (NPU path uses local impl)

torch._dynamo.config.suppress_errors = True

# model_type resolved from --model-path / --model-type at runtime
model_provider = make_model_provider()


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank_idxmap_sft_conv(data_iterator)

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
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat(
            [torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)]
        )
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f"Rank {global_rank}: found NaN in local forward loss calculation. "
            f"Device: {torch.cuda.current_device()}, node: {os.uname()[1]}"
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model: The GPT Model
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
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_online_sft(
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
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_sft_conv(
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

    sft(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=get_patch_args,
    )
