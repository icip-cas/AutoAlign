"""Auto-derive Megatron model config from HuggingFace config.json.

When ``--model-path`` is specified on the CLI, model architecture parameters
(num_layers, hidden_size, ffn_hidden_size, …) are automatically read from the
HF ``config.json``, eliminating the need to hardcode them per model size in
shell scripts.  Explicitly provided CLI arguments always take precedence.

Usage in shell scripts::

    torchrun ... -m autoalign.megatron.entries.sft \
        --model-path /path/to/Qwen2.5-7B-Instruct \
        --load $MCORE_CKPT_PATH \
        --micro-batch-size 4 \
        ...

The following args are auto-derived (when not explicitly provided):

    --num-layers, --hidden-size, --ffn-hidden-size, --num-attention-heads,
    --max-position-embeddings, --norm-epsilon, --rotary-base,
    --attention-dropout, --extra-vocab-size, --num-query-groups,
    --group-query-attention, --untie-embeddings-and-output-weights,
    --swiglu, --normalization, --position-embedding-type,
    --use-rotary-position-embeddings, --disable-bias-linear,
    --add-qkv-bias, --rotary-percent, --patch-tokenizer-type
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---- HF → Megatron field mapping ------------------------------------------
# Key   = Megatron CLI arg name (with underscores, matching argparse dest)
# Value = list of candidate HF config.json keys (first hit wins)
_FIELD_MAPPING: Dict[str, List[str]] = {
    "num_layers": ["num_hidden_layers"],
    "hidden_size": ["hidden_size"],
    "ffn_hidden_size": ["intermediate_size"],
    "num_attention_heads": ["num_attention_heads"],
    "max_position_embeddings": ["max_position_embeddings"],
    "norm_epsilon": ["rms_norm_eps", "layer_norm_epsilon", "layer_norm_eps"],
    "rotary_base": ["rope_theta"],
    "attention_dropout": ["attention_dropout"],
}

# Fields that require value inversion (HF True → Megatron False, etc.)
_INVERTED_BOOL_FIELDS: Dict[str, List[str]] = {
    "untie_embeddings_and_output_weights": ["tie_word_embeddings"],
}

# Per-model-type fixed architectural flags
_MODEL_TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "qwen2": {
        "swiglu": True,
        "normalization": "RMSNorm",
        "position_embedding_type": "rope",
        "use_rotary_position_embeddings": True,
        "disable_bias_linear": True,
        "add_qkv_bias": True,
        "rotary_percent": 1.0,
        "patch_tokenizer_type": "Qwen2Tokenizer",
    },
}


# ---- helpers ---------------------------------------------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _get_tokenizer_vocab_size(model_path: str) -> Optional[int]:
    """Read the base vocab size from ``tokenizer.json`` (if present)."""
    tok_path = os.path.join(model_path, "tokenizer.json")
    if not os.path.isfile(tok_path):
        return None
    try:
        tok_data = _read_json(tok_path)
        vocab = tok_data.get("model", {}).get("vocab")
        if isinstance(vocab, dict):
            return len(vocab)
    except Exception as exc:
        logger.debug("Could not parse tokenizer.json: %s", exc)
    return None


def _match_model_type(model_type: str) -> Optional[str]:
    """Return the key in ``_MODEL_TYPE_DEFAULTS`` that matches *model_type*."""
    for key in _MODEL_TYPE_DEFAULTS:
        if model_type.startswith(key):
            return key
    return None


# ---- public API ------------------------------------------------------------

def derive_megatron_args(model_path: str) -> Dict[str, Any]:
    """Return a dict of Megatron CLI-ready args derived from HF config.

    Keys are argparse *dest* names (underscores), suitable for conversion to
    CLI ``--kebab-case`` names via ``key.replace('_', '-')``.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")

    hf = _read_json(config_path)
    result: Dict[str, Any] = {}

    # 1. Standard field mapping
    for dest, hf_keys in _FIELD_MAPPING.items():
        for hf_key in hf_keys:
            if hf_key in hf and hf[hf_key] is not None:
                val = hf[hf_key]
                if dest == "rotary_base":
                    val = int(val)
                result[dest] = val
                break

    # 2. Inverted boolean fields (e.g. tie_word_embeddings=False → untie=True)
    for dest, hf_keys in _INVERTED_BOOL_FIELDS.items():
        for hf_key in hf_keys:
            if hf_key in hf and hf[hf_key] is not None:
                result[dest] = not hf[hf_key]
                break

    # 3. GQA (group-query-attention)
    num_kv_heads = None
    for hf_key in ["num_key_value_heads"]:
        if hf_key in hf and hf[hf_key] is not None:
            num_kv_heads = hf[hf_key]
            break
    num_attn_heads = result.get("num_attention_heads")
    if num_kv_heads is not None and num_attn_heads is not None:
        if num_kv_heads != num_attn_heads:
            result["group_query_attention"] = True
            result["num_query_groups"] = num_kv_heads

    # 4. Extra vocab size and vocab_size for NullTokenizer
    # HF config.json vocab_size is typically already padded/aligned.
    # NullTokenizer adds +1 (eod), so we subtract 1 to compensate:
    #   NullTokenizer(N-1).vocab_size = N → padded = ceil(N/multiple)*multiple
    # If N is already aligned, padded = N, matching the checkpoint.
    hf_vocab_size = hf.get("vocab_size")
    if hf_vocab_size is not None:
        tok_vocab_size = _get_tokenizer_vocab_size(model_path)
        if tok_vocab_size is not None and tok_vocab_size < hf_vocab_size:
            result["extra_vocab_size"] = hf_vocab_size - tok_vocab_size
        # Subtract 1 because NullTokenizer adds +1 for eod token
        result["vocab_size"] = hf_vocab_size - 1

    # 5. Model-type-specific fixed flags
    model_type = hf.get("model_type", "")
    matched = _match_model_type(model_type)
    if matched:
        for k, v in _MODEL_TYPE_DEFAULTS[matched].items():
            result.setdefault(k, v)

    # 6. Inject model_type itself (normalized to registry key)
    result["model_type_name"] = matched or model_type

    return result


def inject_hf_model_args(parser) -> None:
    """Pre-scan ``sys.argv`` for ``--model-path`` and inject derived args.

    Should be called from ``get_patch_args(parser)`` **after** the
    ``--model-path`` argument has been added to the parser.

    Only args that are *not* already present on the CLI are injected.
    The parser must already contain the target actions (i.e. Megatron's
    standard args must have been registered before this is called).
    """
    # Find --model-path in sys.argv
    model_path = None
    for i, v in enumerate(sys.argv):
        if v == "--model-path" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            break
    if model_path is None:
        return

    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        logger.warning("--model-path %s has no config.json, skipping auto-derivation", model_path)
        return

    try:
        derived = derive_megatron_args(model_path)
    except Exception as exc:
        logger.warning("Failed to derive model config from %s: %s", model_path, exc)
        return

    # Collect all known CLI arg names from the parser
    known_cli_names = set()
    for action in parser._actions:
        known_cli_names.update(action.option_strings)

    injected = []
    for key, value in derived.items():
        cli_name = "--" + key.replace("_", "-")
        # Skip if already on CLI
        if cli_name in sys.argv:
            continue
        # Skip if parser doesn't recognize this arg
        if cli_name not in known_cli_names:
            logger.debug("Skipping unknown arg %s=%s", cli_name, value)
            continue

        if isinstance(value, bool):
            if value:
                sys.argv.append(cli_name)
                injected.append(cli_name)
        else:
            sys.argv.extend([cli_name, str(value)])
            injected.append(f"{cli_name} {value}")

    if injected:
        logger.info("Auto-derived from --model-path %s:\n  %s", model_path, "\n  ".join(injected))
