# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# modified from https://github.com/feifeibear/long-context-attention/blob/main/yunchang/ulysses/attn_layer.py

# Copyright (c) Qihoo 360 Corporation and HaoshengZou
# SPDX-License-Identifier: Apache-2.0
# modified from https://github.com/Qihoo360/360-LLaMA-Factory

import torch

from typing import Any, Optional
from torch import Tensor
import torch.distributed as dist
from autoalign.ulysses.seq_comm import SeqAllToAll4D
import transformers.modeling_flash_attention_utils

from functools import partial
from typing import Dict, Optional, Union, Tuple, Callable, Literal, List, Any
from datasets import Dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
import transformers
from autoalign.data.sequence_parallel import pad_sequence, sp_split
from packaging import version
from functools import lru_cache
import importlib.metadata

class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all. This flag can save cuda memory but will slow down the speed.
        attn_fn (callable): attention type
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        attn_fn: Optional[callable] = None,
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.attn_fn = attn_fn

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        dropout_p=0.0,
        softmax_scale=None,
        position_ids: Optional[torch.Tensor] = None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        context_layer = self.attn_fn(
            q,
            k,
            v,
            attention_mask,
            query_length=query_length,
            is_causal=causal,
            dropout=dropout_p,
            position_ids=position_ids,
            softmax_scale=softmax_scale,  
            softcap=softcap,
            deterministic=deterministic,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync
        )
        # out e.g., [s/p::h]
        return output

def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")

@lru_cache
def is_transformers_version_greater_than(content: str):
    return _get_package_version("transformers") >= version.parse(content)

def get_sequence_parallel_preprocess(
    data_args,
    model_args,
    stage: Literal["pad", "split"],
    tokenizer: AutoTokenizer,
) -> Tuple[Callable, Callable]:
    if stage == "pad":
        preprocess_func = partial(pad_sequence, data_args=data_args, tokenizer=tokenizer)
    elif stage == "split":
        preprocess_func = partial(sp_split, model_args=model_args)
    else:
        raise NotImplementedError(f"Unexpected stage in sequence_parallel_preprocess: {stage}")

    return preprocess_func

def get_sequence_parallel_dataset(
    dataset: Optional[Union[Dataset, IterableDataset]],
    data_args,
    model_args,
    training_args: TrainingArguments,
    tokenizer: AutoTokenizer,
    is_eval: bool = False,
) -> Optional[Union[Dataset, IterableDataset]]:
    kwargs = dict(
        desc="Running padding split on dataset"
    )
    pad_sequence_func = get_sequence_parallel_preprocess(
        data_args=data_args, model_args=model_args, stage="pad", tokenizer=tokenizer
    )
    dataset = dataset.map(
        pad_sequence_func, batched=True, batch_size=data_args.preprocessing_batch_size, **kwargs
    )
    kwargs = dict(
        desc="Running sequence parallel split on dataset"
    )
    sp_dataset_func = get_sequence_parallel_preprocess(
        data_args=data_args, model_args=model_args, stage="split", tokenizer=tokenizer
    )
    dataset = dataset.map(
        sp_dataset_func, batched=True, batch_size=data_args.preprocessing_batch_size, **kwargs
    )
    
    return dataset

def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]

def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    q_len,
    sequence_parallel_size=1,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="ulysses",
    attn_fn=None,
    **kwargs,
):
    if mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_fn=attn_fn)
        attn_output = dist_attn(query_states, key_states, value_states, attention_mask, query_length=q_len * sequence_parallel_size, deterministic=deterministic, dropout_p=dropout, causal=is_causal) # reset query_length to the real q_len before sp, Special settings for ulysses
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output

def apply_sequence_parallel(model_args, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)
    original_attn = transformers.modeling_flash_attention_utils._flash_attention_forward

    try:
        if model_args.sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism, attn_fn=original_attn, sequence_parallel_size=model_args.sequence_parallel_size)
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward

        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
        
        # Store the original method
        original_update_causal_mask = Qwen3Model._update_causal_mask
        
        def new_update_causal_mask(
            self,
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            output_attentions=False,
        ):
            # Fix the condition check for sequence_parallel_attention
            if (self.config._attn_implementation == "flash_attention_2" or 
                self.config._attn_implementation == "sequence_parallel_attention"):
                if attention_mask is not None and past_key_values is not None:
                    is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                    if is_padding_right:
                        raise ValueError(
                            "You are attempting to perform batched generation with padding_side='right'"
                            " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                            " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                        )
                if attention_mask is not None and 0.0 in attention_mask:
                    return attention_mask
                return None
                
            # Continue with the rest of the original method
            return original_update_causal_mask(
                self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions
            )
        
        # Apply the patch
        Qwen3Model._update_causal_mask = new_update_causal_mask
        Qwen2Model._update_causal_mask = new_update_causal_mask
        print("Fixed attention implementation check for sequence_parallel_attention support")

        # AttentionInterface for qwen3 and newer models
        if is_transformers_version_greater_than("4.51.0"):
            from transformers import AttentionInterface

            # modified from integrations/flash_attention.py
            from typing import Optional, Tuple

            import torch

            from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask

            _use_top_left_mask = flash_attn_supports_top_left_mask()

            def sequence_parallel_attention(
                module: torch.nn.Module,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                attention_mask: Optional[torch.Tensor],
                dropout: float = 0.0,
                scaling: Optional[float] = None,
                sliding_window: Optional[int] = None,
                softcap: Optional[float] = None,
                **kwargs,
            ) -> Tuple[torch.Tensor, None]:
                # This is before the transpose
                seq_len = query.shape[2]

                # FA2 uses non-transposed inputs
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                # In PEFT, usually we cast the layer norms in float32 for training stability reasons
                # therefore the input hidden states gets silently casted in float32. Hence, we need
                # cast them back in the correct dtype just to be sure everything works as expected.
                # This might slowdown training & inference so it is recommended to not cast the LayerNorms
                # in fp32. (usually our RMSNorm modules handle it correctly)
                target_dtype = None
                if query.dtype == torch.float32:
                    if torch.is_autocast_enabled():
                        target_dtype = torch.get_autocast_gpu_dtype()
                    # Handle the case where the model is quantized
                    elif hasattr(module.config, "_pre_quantization_dtype"):
                        target_dtype = module.config._pre_quantization_dtype
                    else:
                        target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

                # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
                kwargs.pop("is_causal", None)

                attn_output = new_flash_attention_forward(
                    query,
                    key,
                    value,
                    attention_mask,
                    q_len=seq_len,
                    is_causal=module.is_causal,
                    dropout=dropout,
                    softmax_scale=scaling,
                    sliding_window=sliding_window,
                    softcap=softcap,
                    use_top_left_mask=_use_top_left_mask,
                    target_dtype=target_dtype,
                    **kwargs,
                )

                return attn_output, None


            AttentionInterface.register("sequence_parallel_attention", sequence_parallel_attention)

    except Exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please pip install transformers within the versions that llama-factory requires. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/Qihoo360/360-llama-factory"
        )

    return group_this