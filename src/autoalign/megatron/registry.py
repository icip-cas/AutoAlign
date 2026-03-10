"""Model registration mechanism for Megatron model families.

Provides a registry pattern so that adding a new model family (e.g. LLaMA,
DeepSeek) requires only a single ``register_megatron_model()`` call.

Usage::

    # Registration (typically in a setup module):
    from autoalign.megatron.registry import register_megatron_model, MegatronModelMeta
    register_megatron_model(MegatronModelMeta(
        model_type="qwen2",
        bridge_cls=Qwen2Bridge,
        transformer_config_cls=Qwen2TransformerConfig,
        get_layer_spec_local=get_gpt_layer_local_spec,
        get_layer_spec_te=get_gpt_layer_with_transformer_engine_spec,
        model_cls=GPTModel,
    ))

    # Usage (in training entry points):
    from autoalign.megatron.registry import make_model_provider
    model_provider = make_model_provider("qwen2")           # SFT
    model_provider = make_model_provider("qwen2",           # DPO
        model_cls=GPTModelDPO,
        extra_args_fn=lambda args: dict(beta=args.beta, ...),
    )
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

__all__ = [
    "MegatronModelMeta",
    "register_megatron_model",
    "get_model_meta",
    "make_model_provider",
]


@dataclass
class MegatronModelMeta:
    """Metadata for a registered Megatron model family.

    Attributes:
        model_type: HF model type identifier (e.g. "qwen2").
        bridge_cls: GPTBridge subclass for weight conversion.
        transformer_config_cls: Megatron TransformerConfig subclass.
        get_layer_spec_local: Callable returning a local layer spec.
            Signature: (num_experts, moe_grouped_gemm, qk_layernorm) -> spec
        get_layer_spec_te: Same but for Transformer Engine.
        model_cls: Default GPTModel class for this model family.
    """
    model_type: str
    bridge_cls: type
    transformer_config_cls: type
    get_layer_spec_local: Callable
    get_layer_spec_te: Callable
    model_cls: Optional[type] = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, MegatronModelMeta] = {}
_AUTO_REGISTERED = False


def register_megatron_model(meta: MegatronModelMeta, *, exist_ok: bool = False):
    """Register a Megatron model family.

    Args:
        meta: Model family metadata.
        exist_ok: If False, raise on duplicate registration.
    """
    if not exist_ok and meta.model_type in _REGISTRY:
        raise ValueError(
            f"Model type '{meta.model_type}' is already registered. "
            f"Pass exist_ok=True to override."
        )
    _REGISTRY[meta.model_type] = meta


def get_model_meta(model_type: str) -> MegatronModelMeta:
    """Look up the metadata for a registered model type.

    Triggers lazy auto-registration of built-in model families on first call.
    """
    _ensure_auto_registered()
    model_type = model_type.lower().replace("-", "_")
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[model_type]


def list_registered_models():
    """Return a list of registered model type names."""
    _ensure_auto_registered()
    return list(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Model provider factory
# ---------------------------------------------------------------------------

def make_model_provider(model_type: str = None, model_cls=None, extra_args_fn=None):
    """Create a ``model_provider`` function for Megatron training.

    This replaces the boilerplate model_provider in each training entry point.
    The returned function has the standard Megatron signature::

        model_provider(pre_process=True, post_process=True) -> nn.Module

    Args:
        model_type: Registered model type (e.g. "qwen2").  When ``None``,
            resolved from ``args.model_type`` at call time (requires
            ``--model-type`` or ``--model-path`` on the CLI).
        model_cls: Override the default model class from the registry
            (e.g. GPTModelDPO for DPO training).
        extra_args_fn: Optional callable ``(args) -> dict`` that returns
            additional constructor kwargs (e.g. DPO's beta, loss_type).

    Returns:
        A model_provider function suitable for Megatron's training API.

    Example::

        # SFT: model type resolved from --model-path at runtime
        model_provider = make_model_provider()

        # DPO: uses GPTModelDPO with extra args
        model_provider = make_model_provider(
            model_cls=GPTModelDPO,
            extra_args_fn=lambda args: dict(
                beta=args.beta,
                label_smoothing=args.label_smoothing,
                loss_type=args.loss_type,
            ),
        )
    """
    # If model_type given at definition time, resolve meta eagerly.
    _eager_meta = get_model_meta(model_type) if model_type is not None else None

    def model_provider(pre_process=True, post_process=True):
        from megatron.training import get_args, print_rank_0
        from megatron.training.arguments import core_transformer_config_from_args

        args = get_args()

        # Resolve model type: eager (from arg) or lazy (from CLI --model-type)
        if _eager_meta is not None:
            meta = _eager_meta
            _type_name = model_type
        else:
            _type_name = getattr(args, "model_type_name", None)
            if _type_name is None:
                raise ValueError(
                    "Cannot determine model type. Provide --model-type-name on the "
                    "CLI, or pass model_type= to make_model_provider()."
                )
            meta = get_model_meta(_type_name)

        _model_cls = model_cls or meta.model_cls
        if _model_cls is None:
            raise ValueError(
                f"No model class for '{_type_name}'. "
                f"Pass model_cls= to make_model_provider()."
            )

        print_rank_0(f"building {_type_name} model ...")

        try:
            config = core_transformer_config_from_args(args, meta.transformer_config_cls)
        except TypeError:
            # MindSpeed wraps this function with a 1-arg signature
            config = core_transformer_config_from_args(args)
        use_te = args.transformer_impl == "transformer_engine"

        if use_te:
            print_rank_0(f"building {_type_name} model in TE...")
            spec = meta.get_layer_spec_te(
                args.num_experts, args.moe_grouped_gemm, args.qk_layernorm,
            )
        else:
            print_rank_0(f"building {_type_name} model in Mcore...")
            spec = meta.get_layer_spec_local(
                args.num_experts, args.moe_grouped_gemm, args.qk_layernorm,
            )

        kwargs = dict(
            config=config,
            transformer_layer_spec=spec,
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
        )

        if extra_args_fn is not None:
            kwargs.update(extra_args_fn(args))

        return _model_cls(**kwargs)

    return model_provider


# ---------------------------------------------------------------------------
# Built-in model registrations (lazy)
# ---------------------------------------------------------------------------

def _ensure_auto_registered():
    """Lazily register all built-in model families on first lookup."""
    global _AUTO_REGISTERED
    if _AUTO_REGISTERED:
        return
    _AUTO_REGISTERED = True
    _register_qwen2()


def _register_qwen2():
    """Register the Qwen2 / Qwen2.5 model family.

    Uses stock Megatron-LM APIs (compatible with core_v0.12.1) instead of
    Pai-Megatron-Patch to avoid version conflicts with MindSpeed.
    """
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer import TransformerConfig
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from autoalign.megatron.bridge import Qwen2Bridge

    register_megatron_model(MegatronModelMeta(
        model_type="qwen2",
        bridge_cls=Qwen2Bridge,
        transformer_config_cls=TransformerConfig,
        get_layer_spec_local=get_gpt_layer_local_spec,
        get_layer_spec_te=get_gpt_layer_with_transformer_engine_spec,
        model_cls=GPTModel,
    ))
