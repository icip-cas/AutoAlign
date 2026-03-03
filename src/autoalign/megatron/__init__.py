"""AutoAlign Megatron integration module.

This module provides Megatron-LM based distributed training for LLM alignment.
It requires additional dependencies that must be installed separately.
See docs/megatron.md for installation instructions.

Environment variables:
    MEGATRON_LM_PATH: Path to a local Megatron-LM repository. When set, this
        path is added to sys.path so that ``import megatron`` resolves to the
        specified repo. This makes it easy to swap between different Megatron
        implementations (e.g. NVIDIA GPU vs Ascend NPU).

Usage:
    export MEGATRON_LM_PATH=/path/to/Megatron-LM
    export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
"""

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-configure MEGATRON_LM_PATH (must run before any megatron import)
# ---------------------------------------------------------------------------

_MEGATRON_LM_PATH = os.environ.get("MEGATRON_LM_PATH")

if _MEGATRON_LM_PATH:
    _MEGATRON_LM_PATH = os.path.abspath(_MEGATRON_LM_PATH)
    if os.path.isdir(_MEGATRON_LM_PATH):
        if _MEGATRON_LM_PATH not in sys.path:
            sys.path.insert(0, _MEGATRON_LM_PATH)
            logger.info("MEGATRON_LM_PATH set: added %s to sys.path", _MEGATRON_LM_PATH)
    else:
        logger.warning(
            "MEGATRON_LM_PATH=%s does not exist or is not a directory, ignoring.",
            _MEGATRON_LM_PATH,
        )

# ---------------------------------------------------------------------------
# Dependency availability checks
# ---------------------------------------------------------------------------

def is_megatron_available() -> bool:
    """Check if megatron-core / megatron.training is importable."""
    return (
        importlib.util.find_spec("megatron") is not None
        and importlib.util.find_spec("megatron.core") is not None
    )


def is_megatron_patch_available() -> bool:
    """Check if Pai-Megatron-Patch (megatron_patch) is importable."""
    return importlib.util.find_spec("megatron_patch") is not None


def is_apex_available() -> bool:
    """Check if NVIDIA Apex is importable."""
    return importlib.util.find_spec("apex") is not None


def is_flash_attn_available() -> bool:
    """Check if Flash Attention is importable."""
    return importlib.util.find_spec("flash_attn") is not None


def is_transformer_engine_available() -> bool:
    """Check if NVIDIA Transformer Engine is importable."""
    return importlib.util.find_spec("transformer_engine") is not None


_INSTALL_GUIDE = "See docs/megatron.md for installation instructions."


def require_megatron():
    """Raise a clear error if megatron-core is not installed."""
    if not is_megatron_available():
        raise ImportError(
            "megatron-core is required but not installed. " + _INSTALL_GUIDE
        )


def require_megatron_patch():
    """Raise a clear error if Pai-Megatron-Patch is not installed."""
    if not is_megatron_patch_available():
        raise ImportError(
            "Pai-Megatron-Patch (megatron_patch) is required but not installed. "
            + _INSTALL_GUIDE
        )


def get_megatron_version() -> str:
    """Return the megatron-core version string, or 'not installed'."""
    try:
        import megatron.core
        return getattr(megatron.core, "__version__", "unknown")
    except ImportError:
        return "not installed"


# ---------------------------------------------------------------------------
# Log status on import (non-fatal — allows lint / static analysis to pass)
# ---------------------------------------------------------------------------

_deps = {
    "megatron-core": is_megatron_available(),
    "megatron_patch": is_megatron_patch_available(),
    "apex": is_apex_available(),
    "flash-attn": is_flash_attn_available(),
    "transformer-engine": is_transformer_engine_available(),
}

_missing = [k for k, v in _deps.items() if not v]
if _missing:
    logger.debug(
        "autoalign.megatron: optional dependencies not found: %s. "
        "Megatron training features will not be available until they are installed.",
        ", ".join(_missing),
    )
