"""Bridge abstraction for HF <-> Megatron-Core weight conversion.

Provides a clean, declarative approach to bidirectional weight conversion
between HuggingFace and Megatron-Core model formats, including TP/PP/EP
checkpoint sharding.

Inspired by ms-swift's GPTBridge pattern.  Subclass ``GPTBridge`` and override
``_convert_layer_hf_to_mg`` / ``_convert_layer_mg_to_hf`` to support a new
model family.  Currently provides ``Qwen2Bridge`` for Qwen2 / Qwen2.5 models.

Usage::

    bridge = get_bridge("qwen2")
    bridge.hf_to_mg(hf_model, mg_model, args)   # HF -> Megatron
    bridge.mg_to_hf(mg_model, hf_model, args)   # Megatron -> HF
"""

import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from collections.abc import Mapping, Sequence

__all__ = ["GPTBridge", "Qwen2Bridge", "get_bridge"]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _deep_getattr(obj, dotted_path):
    """Traverse nested attributes via a dotted path string."""
    for attr in dotted_path.split("."):
        obj = getattr(obj, attr)
    return obj


@torch.inference_mode()
def clone_state_dict(elem):
    """Deep-clone all tensors in a nested structure."""
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return elem.clone()
    elif isinstance(elem, (np.ndarray, str)):
        return elem
    elif isinstance(elem, Mapping):
        return elem_type({k: clone_state_dict(v) for k, v in elem.items()})
    elif isinstance(elem, Sequence):
        return elem_type([clone_state_dict(v) for v in elem])
    return elem


def _extract_expert_rank(key):
    """Extract local expert rank from ``...local_experts.3.linear_fc1...``."""
    match = re.search(r'local_experts\.(\d+)\.', key)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Base Bridge
# ---------------------------------------------------------------------------

class GPTBridge:
    """Base bridge for bidirectional HF <-> Megatron-Core weight conversion.

    A bridge handles two concerns:

    1. **Format conversion**: weight key remapping and tensor transformations
       (QKV merge/split, gate+up merge/split) between HF and Megatron naming.
    2. **TP/PP/EP sharding**: splitting and gathering Megatron state-dict
       tensors for distributed checkpoint saving and loading.

    Subclass and override ``_convert_layer_hf_to_mg`` and
    ``_convert_layer_mg_to_hf`` for your model family.
    """

    # --- HF model structure (override in subclass if needed) ---
    hf_layers_prefix: str = "model.layers"
    hf_embed_key: str = "model.embed_tokens.weight"
    hf_final_norm_key: str = "model.norm.weight"
    hf_lm_head_key: str = "lm_head.weight"

    # ================================================================== #
    #  Format Conversion: HF model <-> Megatron model
    # ================================================================== #

    def hf_to_mg(self, hf_model, mg_model, args):
        """Copy weights from an HF model into a Megatron model (in-place)."""
        if args.fp16:
            mg_model = mg_model.half()
            hf_model = hf_model.half()
        elif args.bf16:
            mg_model = mg_model.bfloat16()
            hf_model = hf_model.bfloat16()

        with torch.no_grad():
            # Embedding
            mg_model.embedding.word_embeddings.weight.copy_(
                _deep_getattr(hf_model, self.hf_embed_key)
            )
            # Transformer layers
            hf_layers = _deep_getattr(hf_model, self.hf_layers_prefix)
            for mglayer, hflayer in zip(mg_model.decoder.layers, hf_layers):
                self._convert_layer_hf_to_mg(hflayer, mglayer, args)
            # Final LayerNorm
            mg_model.decoder.final_layernorm.weight.copy_(
                _deep_getattr(hf_model, self.hf_final_norm_key)
            )
            # LM head (only when untied)
            if args.untie_embeddings_and_output_weights:
                mg_model.output_layer.weight.copy_(
                    _deep_getattr(hf_model, self.hf_lm_head_key)
                )

    def mg_to_hf(self, mg_model, hf_model, args):
        """Copy weights from a Megatron model into an HF model (in-place)."""
        if args.fp16:
            mg_model = mg_model.half()
            hf_model = hf_model.half()
        elif args.bf16:
            mg_model = mg_model.bfloat16()
            hf_model = hf_model.bfloat16()

        with torch.no_grad():
            # Embedding
            _deep_getattr(hf_model, self.hf_embed_key).copy_(
                mg_model.embedding.word_embeddings.weight
            )
            # Transformer layers
            hf_layers = _deep_getattr(hf_model, self.hf_layers_prefix)
            for mglayer, hflayer in zip(mg_model.decoder.layers, hf_layers):
                self._convert_layer_mg_to_hf(mglayer, hflayer, args)
            # Final LayerNorm
            _deep_getattr(hf_model, self.hf_final_norm_key).copy_(
                mg_model.decoder.final_layernorm.weight
            )
            # LM head
            if args.untie_embeddings_and_output_weights:
                _deep_getattr(hf_model, self.hf_lm_head_key).copy_(
                    mg_model.output_layer.weight
                )

    def _convert_layer_hf_to_mg(self, hflayer, mglayer, args):
        """Convert a single transformer layer HF -> Megatron.

        Must be overridden in subclass.
        """
        raise NotImplementedError

    def _convert_layer_mg_to_hf(self, mglayer, hflayer, args):
        """Convert a single transformer layer Megatron -> HF.

        Must be overridden in subclass.
        """
        raise NotImplementedError

    # ================================================================== #
    #  TP Split / Gather (Megatron state-dict level)
    # ================================================================== #

    def split_tensor_for_tp(self, key, tensor, tp_rank, tp_size, args):
        """Split a Megatron state-dict tensor for the given TP rank.

        Returns the TP-sliced tensor, or the original if no splitting applies.
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor

        head_dim = args.hidden_size // args.num_attention_heads
        group_per_split = args.num_query_groups // tp_size

        # --- QKV ---
        if 'linear_qkv.weight' in key:
            viewed = tensor.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            return viewed[group_per_split * tp_rank:group_per_split * (tp_rank + 1)].view(-1, args.hidden_size)

        if 'linear_qkv.bias' in key:
            viewed = tensor.view(args.num_query_groups, -1)
            return viewed[group_per_split * tp_rank:group_per_split * (tp_rank + 1)].view(-1)

        # --- Row-parallel: O-proj and non-expert down-proj ---
        if 'linear_proj' in key or ('linear_fc2' in key and 'local_experts' not in key and 'shared_expert' not in key):
            seg = tensor.shape[1] // tp_size
            return tensor[:, seg * tp_rank:seg * (tp_rank + 1)]

        # --- Embeddings / output layer ---
        if 'embedding' in key or 'output_layer' in key:
            seg = tensor.shape[0] // tp_size
            return tensor[seg * tp_rank:seg * (tp_rank + 1)]

        # --- Column-parallel: non-expert MLP fc1 ---
        if 'linear_fc1' in key and 'norm' not in key and 'local_experts' not in key and 'shared_expert' not in key:
            ffn = args.ffn_hidden_size
            viewed = tensor.view(-1, ffn, args.hidden_size)
            seg = ffn // tp_size
            return viewed[:, seg * tp_rank:seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)

        # --- MoE expert layers ---
        if 'local_experts' in key:
            if 'linear_fc1' in key and 'norm' not in key:
                ffn = args.moe_ffn_hidden_size
                viewed = tensor.view(-1, ffn, args.hidden_size)
                seg = ffn // tp_size
                return viewed[:, seg * tp_rank:seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
            if 'linear_fc2' in key:
                seg = tensor.shape[1] // tp_size
                return tensor[:, seg * tp_rank:seg * (tp_rank + 1)]

        # --- Shared expert layers ---
        if 'shared_expert' in key and 'gate' not in key:
            if 'linear_fc1' in key:
                ffn = args.shared_moe_ffn_hidden_size
                viewed = tensor.view(-1, ffn, args.hidden_size)
                seg = ffn // tp_size
                return viewed[:, seg * tp_rank:seg * (tp_rank + 1), :].reshape(-1, args.hidden_size)
            if 'linear_fc2' in key:
                seg = tensor.shape[1] // tp_size
                return tensor[:, seg * tp_rank:seg * (tp_rank + 1)]

        # No split needed (norms, router weights, etc.)
        return tensor

    def gather_tensor_from_tp(self, key, shards, args):
        """Gather TP-split shards for a single key into a full tensor."""
        if not isinstance(shards[0], torch.Tensor):
            return shards[0]

        # Scalars that are replicated, not split
        if 'norm' in key or 'router' in key or ('gate' in key and 'linear' not in key):
            return shards[0]

        head_dim = args.hidden_size // args.num_attention_heads
        tp_size = len(shards)
        group_per_split = args.num_query_groups // tp_size

        if 'linear_qkv.weight' in key:
            viewed = [x.view(group_per_split, -1, head_dim, args.hidden_size) for x in shards]
            return torch.cat(viewed, dim=0).view(-1, args.hidden_size)

        if 'linear_qkv.bias' in key:
            viewed = [x.view(group_per_split, -1) for x in shards]
            return torch.cat(viewed, dim=0).view(-1)

        if 'embedding' in key or 'output_layer' in key:
            return torch.cat(shards, dim=0)

        if 'linear_proj' in key or 'linear_fc2' in key:
            return torch.cat(shards, dim=1)

        if 'linear_fc1' in key and 'norm' not in key:
            viewed = [x.view(2, -1, args.hidden_size) for x in shards]
            return torch.cat(viewed, dim=1).view(-1, args.hidden_size)

        return shards[0]

    # ================================================================== #
    #  Megatron Checkpoint Save / Load (TP + PP + EP)
    # ================================================================== #

    def load_mg_state_dict(self, args) -> Dict[str, torch.Tensor]:
        """Load Megatron checkpoint shards and gather into a full state dict.

        Handles all combinations of TP, PP, and EP parallelism.
        """
        from megatron.training.checkpointing import (
            get_checkpoint_name, get_checkpoint_tracker_filename, read_metadata,
        )

        model_path = args.load
        tracker_filename = get_checkpoint_tracker_filename(model_path)
        iteration, release = read_metadata(tracker_filename)

        tp_size = args.target_tensor_model_parallel_size
        pp_size = args.target_pipeline_model_parallel_size
        ep_size = getattr(args, 'target_expert_model_parallel_size', 1)
        has_experts = args.num_experts is not None
        num_layers_per_pp = args.num_layers // pp_size if pp_size > 1 else args.num_layers
        num_local_experts = (args.num_experts // ep_size) if has_experts else 0

        mid_state = defaultdict(list)

        for tp_rank in range(tp_size):
            for ep_rank in range(ep_size if has_experts else 1):
                for pp_rank in range(pp_size):
                    checkpoint_name = self._checkpoint_name(
                        get_checkpoint_name, model_path, iteration, release,
                        tp_rank, tp_size, pp_rank, pp_size,
                        ep_rank, ep_size, has_experts,
                    )
                    print(f'load {checkpoint_name}')
                    import argparse
                    from megatron.core.transformer.enums import AttnBackend
                    torch.serialization.add_safe_globals([argparse.Namespace, AttnBackend])
                    split_state = torch.load(
                        checkpoint_name, map_location="cpu",
                    )['model']

                    for k, v in split_state.items():
                        # PP: remap local layer index -> global
                        if pp_size > 1:
                            match = re.search(r'decoder\.layers\.(\d+)', k)
                            if match:
                                local_layer = int(match.group(1))
                                global_layer = local_layer + pp_rank * num_layers_per_pp
                                k = k.replace(
                                    f'decoder.layers.{local_layer}',
                                    f'decoder.layers.{global_layer}',
                                )

                        # EP: remap local expert index -> global
                        if has_experts and 'local_experts' in k and 'norm' not in k:
                            local_rank = _extract_expert_rank(k)
                            global_rank = local_rank + num_local_experts * ep_rank
                            k = k.replace(
                                f'local_experts.{local_rank}',
                                f'local_experts.{global_rank}',
                            )
                            mid_state[k].append(v)
                        elif ep_rank == 0:
                            mid_state[k].append(v)

        # Gather TP shards
        state_dict = {}
        for k, shards in mid_state.items():
            if tp_size > 1:
                state_dict[k] = self.gather_tensor_from_tp(k, shards, args)
            else:
                state_dict[k] = shards[0]

        return state_dict

    def save_mg_checkpoint(self, mg_model, args):
        """Save a Megatron model to sharded checkpoint files.

        Handles all combinations of TP, PP, and EP parallelism.
        """
        from megatron.training.checkpointing import get_checkpoint_name

        tp_size = args.target_tensor_model_parallel_size
        pp_size = args.target_pipeline_model_parallel_size
        ep_size = getattr(args, 'target_expert_model_parallel_size', 1)
        has_experts = args.num_experts is not None
        num_layers_per_pp = args.num_layers // pp_size
        num_local_experts = (args.num_experts // ep_size) if has_experts else 0

        # Update args for consistency
        args.tensor_model_parallel_size = tp_size
        args.pipeline_model_parallel_size = pp_size
        if has_experts:
            args.expert_model_parallel_size = ep_size
        if tp_size > 1:
            args.sequence_parallel = True

        os.makedirs(args.save, exist_ok=True)

        # Copy tokenizer / config files
        src = args.hf_ckpt_path if hasattr(args, 'hf_ckpt_path') and args.hf_ckpt_path else args.load
        for pattern in ["config*.json", "generation_config.json", "tokenizer*",
                        "vocab.json", "merges.txt"]:
            os.system(f"cp -rf {src}/{pattern} {args.save} 2>/dev/null")
            if hasattr(args, 'load') and args.load != src:
                os.system(f"cp -rf {args.load}/{pattern} {args.save} 2>/dev/null")

        tracker_filepath = os.path.join(args.save, 'latest_checkpointed_iteration.txt')
        with open(tracker_filepath, "w") as f:
            f.write("release")

        full_model = mg_model.state_dict_for_save_checkpoint()
        expert_pattern = re.compile(r'local_experts\.(\d+)\.')

        for tp_rank in range(tp_size):
            for ep_rank in range(ep_size if has_experts else 1):
                for pp_rank in range(pp_size):
                    model_split = {}
                    layer_offset = pp_rank * num_layers_per_pp

                    # PP: determine which global layer IDs belong to this rank
                    pp_global_ids = set()
                    global_to_local = {}
                    for local_id in range(num_layers_per_pp):
                        global_id = local_id + layer_offset
                        pp_global_ids.add(f"decoder.layers.{global_id}")
                        global_to_local[f"decoder.layers.{global_id}"] = local_id

                    for k, v in full_model.items():
                        new_k = k

                        # --- PP filtering and layer remapping ---
                        if pp_size > 1:
                            layer_match = re.match(r'(decoder\.layers\.\d+)', k)
                            if layer_match:
                                layer_prefix = layer_match.group(1)
                                if layer_prefix not in pp_global_ids:
                                    continue
                                local_id = global_to_local[layer_prefix]
                                new_k = k.replace(layer_prefix, f'decoder.layers.{local_id}')
                            elif 'word_embeddings' in k:
                                if pp_rank != 0:
                                    continue
                            elif 'output_layer' in k or 'final_layernorm' in k:
                                if pp_rank != pp_size - 1:
                                    continue
                            else:
                                continue

                        # --- EP filtering and expert remapping ---
                        if has_experts and 'local_experts' in new_k:
                            m = expert_pattern.search(new_k)
                            if m:
                                expert_rank = int(m.group(1))
                                if ep_size > 1:
                                    if expert_rank // num_local_experts != ep_rank:
                                        continue
                                    local_expert = expert_rank % num_local_experts
                                    new_k = new_k.replace(
                                        f'local_experts.{expert_rank}',
                                        f'local_experts.{local_expert}',
                                    )

                        # --- TP splitting ---
                        if tp_size > 1:
                            target_v = self.split_tensor_for_tp(new_k, v, tp_rank, tp_size, args)
                        else:
                            target_v = v

                        model_split[new_k] = target_v

                    checkpoint_name = self._checkpoint_name(
                        get_checkpoint_name, args.save, 0, True,
                        tp_rank, tp_size, pp_rank, pp_size,
                        ep_rank, ep_size, has_experts,
                    )
                    _save_checkpoint_shard(args, model_split, checkpoint_name)

        print(f'megatron model is saved to {args.save}')

    @staticmethod
    def _checkpoint_name(get_checkpoint_name_fn, path, iteration, release,
                         tp_rank, tp_size, pp_rank, pp_size,
                         ep_rank, ep_size, has_experts):
        """Build a Megatron checkpoint file path for a given rank combination."""
        use_pp = pp_size > 1
        args = [path, iteration, release]
        args.append(True if use_pp else None)       # pipeline_parallel
        args.append(tp_rank if tp_size > 1 else None)  # tensor_rank
        args.append(pp_rank if use_pp else None)     # pipeline_rank

        if has_experts:
            if ep_size > 1:
                args.extend([True, ep_rank])
            else:
                args.append(False)
        else:
            args.extend([None, None])

        return get_checkpoint_name_fn(*args)


# ---------------------------------------------------------------------------
# Qwen2 / Qwen2.5 Bridge
# ---------------------------------------------------------------------------

class Qwen2Bridge(GPTBridge):
    """Bridge for the Qwen2 / Qwen2.5 model family.

    Handles:
    - QKV bias (Qwen2 uses bias in Q/K/V projections)
    - SwiGLU MLP (gate_proj + up_proj merged into linear_fc1)
    - MoE experts and shared experts (Qwen2.5-MoE)
    - Transformer-Engine fused-LayerNorm variant
    """

    def _convert_layer_hf_to_mg(self, hflayer, mglayer, args):
        use_te = args.transformer_impl == "transformer_engine"
        num_query_groups = args.num_query_groups
        hidden_size = args.hidden_size
        head_dim = hidden_size // args.num_attention_heads

        # --- Input LayerNorm ---
        if use_te:
            mglayer.self_attention.linear_qkv.layer_norm_weight.copy_(
                hflayer.input_layernorm.weight
            )
        else:
            mglayer.input_layernorm.weight.copy_(hflayer.input_layernorm.weight)

        # --- QKV merge (interleaved by query groups) ---
        q_w = hflayer.self_attn.q_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        k_w = hflayer.self_attn.k_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        v_w = hflayer.self_attn.v_proj.weight.view(num_query_groups, -1, head_dim, hidden_size)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=1).view(-1, hidden_size).contiguous()
        mglayer.self_attention.linear_qkv.weight.copy_(qkv_w)

        q_b = hflayer.self_attn.q_proj.bias.view(num_query_groups, -1)
        k_b = hflayer.self_attn.k_proj.bias.view(num_query_groups, -1)
        v_b = hflayer.self_attn.v_proj.bias.view(num_query_groups, -1)
        qkv_b = torch.cat([q_b, k_b, v_b], dim=1).view(-1).contiguous()
        mglayer.self_attention.linear_qkv.bias.copy_(qkv_b)

        # --- O projection ---
        mglayer.self_attention.linear_proj.weight.copy_(
            hflayer.self_attn.o_proj.weight
        )

        # --- MLP ---
        if args.num_experts is None:
            fc1 = torch.cat([hflayer.mlp.gate_proj.weight,
                             hflayer.mlp.up_proj.weight])
            mglayer.mlp.linear_fc1.weight.copy_(fc1)
            mglayer.mlp.linear_fc2.weight.copy_(hflayer.mlp.down_proj.weight)
        else:
            # MoE router
            mglayer.mlp.router.weight.copy_(hflayer.mlp.gate.weight)
            # Per-expert
            for hf_expert, mg_expert in zip(
                hflayer.mlp.experts, mglayer.mlp.experts.local_experts
            ):
                fc1 = torch.cat([hf_expert.gate_proj.weight,
                                 hf_expert.up_proj.weight])
                mg_expert.linear_fc1.weight.copy_(fc1)
                mg_expert.linear_fc2.weight.copy_(hf_expert.down_proj.weight)
            # Shared expert
            mglayer.mlp.shared_expert_gate.weight.copy_(
                hflayer.mlp.shared_expert_gate.weight
            )
            shared_fc1 = torch.cat([
                hflayer.mlp.shared_expert.gate_proj.weight,
                hflayer.mlp.shared_expert.up_proj.weight,
            ])
            mglayer.mlp.shared_expert.linear_fc1.weight.copy_(shared_fc1)
            mglayer.mlp.shared_expert.linear_fc2.weight.copy_(
                hflayer.mlp.shared_expert.down_proj.weight
            )

        # --- Post-attention LayerNorm ---
        if use_te and not args.num_experts:
            mglayer.mlp.linear_fc1.layer_norm_weight.copy_(
                hflayer.post_attention_layernorm.weight
            )
        else:
            mglayer.pre_mlp_layernorm.weight.copy_(
                hflayer.post_attention_layernorm.weight
            )

    def _convert_layer_mg_to_hf(self, mglayer, hflayer, args):
        use_te = args.transformer_impl == "transformer_engine"
        num_query_groups = args.num_query_groups
        hidden_size = args.hidden_size
        head_dim = hidden_size // args.num_attention_heads
        value_num_per_group = args.num_attention_heads // num_query_groups
        q_dim_per_group = hidden_size // num_query_groups
        kv_dim_per_group = head_dim

        # --- Input LayerNorm ---
        if use_te:
            hflayer.input_layernorm.weight.copy_(
                mglayer.self_attention.linear_qkv.layer_norm_weight
            )
        else:
            hflayer.input_layernorm.weight.copy_(mglayer.input_layernorm.weight)

        # --- QKV split ---
        qkv_w = mglayer.self_attention.linear_qkv.weight.view(
            num_query_groups, -1, head_dim, hidden_size
        )
        q_w, k_w, v_w = torch.split(
            qkv_w, [value_num_per_group, 1, 1], dim=1
        )
        hflayer.self_attn.q_proj.weight.copy_(q_w.reshape(-1, hidden_size))
        hflayer.self_attn.k_proj.weight.copy_(k_w.reshape(-1, hidden_size))
        hflayer.self_attn.v_proj.weight.copy_(v_w.reshape(-1, hidden_size))

        qkv_b = mglayer.self_attention.linear_qkv.bias.view(
            num_query_groups, -1
        )
        q_b, k_b, v_b = torch.split(
            qkv_b, [q_dim_per_group, kv_dim_per_group, kv_dim_per_group], dim=1
        )
        hflayer.self_attn.q_proj.bias.copy_(q_b.contiguous().view(-1))
        hflayer.self_attn.k_proj.bias.copy_(k_b.contiguous().view(-1))
        hflayer.self_attn.v_proj.bias.copy_(v_b.contiguous().view(-1))

        # --- O projection ---
        hflayer.self_attn.o_proj.weight.copy_(
            mglayer.self_attention.linear_proj.weight
        )

        # --- MLP ---
        if args.num_experts is None:
            gate_w, up_w = torch.split(
                mglayer.mlp.linear_fc1.weight, args.ffn_hidden_size
            )
            hflayer.mlp.gate_proj.weight.copy_(gate_w)
            hflayer.mlp.up_proj.weight.copy_(up_w)
            hflayer.mlp.down_proj.weight.copy_(mglayer.mlp.linear_fc2.weight)
        else:
            hflayer.mlp.gate.weight.copy_(mglayer.mlp.router.weight)
            for mg_expert, hf_expert in zip(
                mglayer.mlp.experts.local_experts, hflayer.mlp.experts
            ):
                gate_w, up_w = torch.split(
                    mg_expert.linear_fc1.weight, args.moe_ffn_hidden_size
                )
                hf_expert.gate_proj.weight.copy_(gate_w)
                hf_expert.up_proj.weight.copy_(up_w)
                hf_expert.down_proj.weight.copy_(mg_expert.linear_fc2.weight)
            # Shared expert
            hflayer.mlp.shared_expert_gate.weight.copy_(
                mglayer.mlp.shared_expert_gate.weight
            )
            shared_gate, shared_up = torch.split(
                mglayer.mlp.shared_expert.linear_fc1.weight,
                args.shared_moe_ffn_hidden_size,
            )
            hflayer.mlp.shared_expert.gate_proj.weight.copy_(shared_gate)
            hflayer.mlp.shared_expert.up_proj.weight.copy_(shared_up)
            hflayer.mlp.shared_expert.down_proj.weight.copy_(
                mglayer.mlp.shared_expert.linear_fc2.weight
            )

        # --- Post-attention LayerNorm ---
        if use_te and not args.num_experts:
            hflayer.post_attention_layernorm.weight.copy_(
                mglayer.mlp.linear_fc1.layer_norm_weight
            )
        else:
            hflayer.post_attention_layernorm.weight.copy_(
                mglayer.pre_mlp_layernorm.weight
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BRIDGE_REGISTRY = {
    "qwen2": Qwen2Bridge,
    "qwen2_5": Qwen2Bridge,
}


def get_bridge(model_type: str = "qwen2") -> GPTBridge:
    """Get a bridge instance for the given model type.

    Args:
        model_type: Model family name (e.g. "qwen2", "qwen2_5").

    Returns:
        A ``GPTBridge`` subclass instance.
    """
    model_type = model_type.lower().replace("-", "_")
    if model_type not in _BRIDGE_REGISTRY:
        raise ValueError(
            f"No bridge registered for model type '{model_type}'. "
            f"Available: {list(_BRIDGE_REGISTRY.keys())}"
        )
    return _BRIDGE_REGISTRY[model_type]()


# ---------------------------------------------------------------------------
# Checkpoint I/O helpers
# ---------------------------------------------------------------------------

def _save_checkpoint_shard(args, model_state, checkpoint_name):
    """Save a single Megatron checkpoint shard."""
    state_dict = {
        'args': args,
        'checkpoint_version': 3.0,
        'iteration': 0,
        'model': model_state,
    }
    os.makedirs(os.path.dirname(checkpoint_name), exist_ok=True)
    print(f'save model part {checkpoint_name}')
    torch.save(clone_state_dict(state_dict), checkpoint_name)
