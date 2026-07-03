# Copyright (c) 2024 AutoAlign Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared runtime patches for Megatron entry points."""

from __future__ import annotations

import os

_PATCHES_APPLIED = False
_swanlab_run = None
_swanlab_initialized = False


def apply_entry_patches() -> None:
    """Install compatibility patches used by native Megatron entries."""
    global _PATCHES_APPLIED
    if _PATCHES_APPLIED:
        return
    _PATCHES_APPLIED = True

    # CP NCCL warmup.
    import autoalign.megatron.patch.core.context_parallel  # noqa: F401

    # Patch 1: strict=False — tolerate missing/unexpected keys across impl switches.
    import megatron.training.checkpointing as _ckpt
    import megatron.training.training as _training

    _orig_load_checkpoint = _ckpt.load_checkpoint

    def _patched_load_checkpoint(*args, strict=True, **kwargs):
        return _orig_load_checkpoint(*args, strict=False, **kwargs)

    _ckpt.load_checkpoint = _patched_load_checkpoint
    _training.load_checkpoint = _patched_load_checkpoint

    # Patch 2: SwanLab / WandB logging — inject into Megatron's training_log.
    # SwanLab is lazily initialized on the first training_log call (after
    # torch.distributed is ready).
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
        from megatron.training import get_args
        import torch
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

    # Patch 3: skip empty TE _extra_state (zero bytes → EOFError in pickle.loads).
    # TE _extra_state is only populated after the first forward pass (quantization
    # calibration), so freshly converted or newly created checkpoints always have
    # empty _extra_state tensors. This is expected; just skip them on load.
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


def _maybe_init_swanlab():
    global _swanlab_run, _swanlab_initialized
    if _swanlab_initialized:
        return _swanlab_run
    _swanlab_initialized = True
    try:
        import torch
        from megatron.training import get_args
        args = get_args()
        if 'swanlab' not in getattr(args, 'report_to', []):
            return None
        if torch.distributed.get_rank() != args.world_size - 1:
            return None
        import swanlab
        api_key = os.environ.get('SWANLAB_API_KEY')
        if api_key:
            swanlab.login(api_key=api_key)
        _swanlab_run = swanlab.init(config=vars(args))
        print(f'SwanLab initialized: {_swanlab_run}', flush=True)
    except Exception as e:
        print(f'WARNING: SwanLab init failed ({e}), disabling SwanLab logging', flush=True)
    return _swanlab_run
