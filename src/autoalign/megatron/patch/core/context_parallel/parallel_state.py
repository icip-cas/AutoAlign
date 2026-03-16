"""Patch: eager NCCL communicator warmup for context parallelism.

Problem
-------
Standard Megatron-LM assumes pre-tokenized MMap datasets whose per-rank
loading is near-instantaneous.  With online tokenization, CP ranks load and
tokenize sequences of varying length independently, introducing variable CPU
latency.  CP=0 ranks can therefore reach ring-attention and attempt to lazily
initialize the CP NCCL communicator *before* CP=1 ranks arrive — causing a
permanent deadlock.

Root cause: PyTorch's ProcessGroupNCCL initializes communicators lazily (the
actual ``ncclCommInitRank`` is deferred until the first collective on that
group).  When CP ranks arrive at ring-attention at different times, CP=0 tries
to create the communicator (which requires all CP ranks to participate
simultaneously), while CP=1 is still CPU-bound — deadlock.

Fix
---
Wrap ``parallel_state.initialize_model_parallel`` to call
``dist.barrier(group=g)`` on every process group immediately after they are
created.  The barrier triggers ``ncclCommInitRank`` for each group while all
ranks are still in the initialization phase and guaranteed to be synchronized,
so no rank can race ahead into training communication.
"""

import torch.distributed as dist
from megatron.core import parallel_state as _ps

_patched = False


def apply_cp_nccl_warmup_patch() -> None:
    """Monkey-patch ``parallel_state.initialize_model_parallel`` to add
    eager NCCL communicator initialization for all process groups.

    Idempotent — calling this multiple times has no effect.
    """
    global _patched
    if _patched:
        return
    _patched = True

    _orig = _ps.initialize_model_parallel

    def _patched_init_mp(*args, **kwargs):
        result = _orig(*args, **kwargs)
        _warmup_nccl_groups()
        return result

    _ps.initialize_model_parallel = _patched_init_mp


def _warmup_nccl_groups() -> None:
    """Call ``dist.barrier`` on every process group to force eager NCCL init."""
    for _group_fn in (
        _ps.get_tensor_model_parallel_group,
        _ps.get_data_parallel_group,
        _ps.get_context_parallel_group,
        _ps.get_pipeline_model_parallel_group,
    ):
        try:
            g = _group_fn()
            if g is not None:
                dist.barrier(group=g, device_ids=[torch.cuda.current_device()])
        except Exception:
            pass
