"""DPO checkpoint conversion — delegates to the common conversion logic.

The DPO model has the same weight structure as the base model (DPO is a
training method, not a different architecture), so the same bridge and
conversion pipeline applies.

Usage::

    torchrun ... -m autoalign.megatron.toolkits.checkpoint.qwen.dpo \\
        --model-path <HF_PATH> ...
"""

from .common import main

if __name__ == "__main__":
    main()
