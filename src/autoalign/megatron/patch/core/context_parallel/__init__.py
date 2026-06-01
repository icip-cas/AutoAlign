"""Context-parallel patches for Megatron-LM.

Importing this module applies all patches in this package.  Import it once
before ``pretrain()`` / ``initialize_megatron()`` is called.
"""

from autoalign.megatron.patch.core.context_parallel.parallel_state import (  # noqa: F401
    apply_cp_nccl_warmup_patch,
)

apply_cp_nccl_warmup_patch()
