"""Checkpoint conversion between HuggingFace and Megatron-Core formats.

Supports both directions:
  - HF -> Megatron: ``python -m autoalign.megatron.toolkits.checkpoint.qwen.common``
  - Megatron -> HF: add ``--convert-checkpoint-from-megatron-to-transformers``

Uses the Bridge abstraction (``autoalign.megatron.bridge``) for clean weight
conversion and TP/PP/EP sharding.
"""

import autoalign.megatron  # noqa: F401  # bootstrap MEGATRON_LM_PATH + MindSpeed adaptor before megatron imports

import os
import json
import copy
import safetensors
from functools import partial

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME
try:
    from transformers.modeling_utils import shard_checkpoint
except ImportError:
    # shard_checkpoint removed in transformers >= 4.44; use huggingface_hub instead
    from huggingface_hub.serialization import split_torch_state_dict_into_shards as _split

    def shard_checkpoint(state_dict, max_shard_size="10GB", weights_name=WEIGHTS_NAME):
        shards_iter = _split(state_dict, max_shard_size=max_shard_size, filename_pattern=weights_name.replace(".bin", "_{index}.bin").replace(".safetensors", "_{index}.safetensors"))
        shards = {f: {k: state_dict[k] for k in shard_keys} for f, shard_keys in shards_iter.filename_to_tensors.items()}
        index = shards_iter.metadata if len(shards) > 1 else None
        return shards, index

from megatron.training.initialize import initialize_megatron
from megatron.training import get_args
from megatron.training.utils import get_ltor_masks_and_position_ids

from megatron_patch.arguments import get_patch_args as get_patch_args_pai

from autoalign.megatron.bridge import Qwen2Bridge, clone_state_dict
from autoalign.megatron.registry import make_model_provider
from autoalign.megatron.patch.arguments import get_patch_args as get_patch_args_autoalign

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


# ---------------------------------------------------------------------------
# CLI argument helpers
# ---------------------------------------------------------------------------

def add_model_args(parser):
    parser.add_argument("--target-tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--target-pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--target-expert-model-parallel-size", type=int, default=1)
    parser.add_argument("--hf-ckpt-path", type=str)
    parser.add_argument("--save-safetensors", action='store_false')
    parser.add_argument("--test-convert-precision", action='store_true',
                        help="Run HF vs Megatron forward comparison after conversion to verify precision. "
                             "Requires enough device memory to hold both models simultaneously.")
    return parser


def add_extra_args(parser):
    parser.conflict_handler = 'resolve'
    parser = get_patch_args_pai(parser)
    parser = get_patch_args_autoalign(parser)
    parser = add_model_args(parser)
    # torchrun sets LOCAL_RANK env var but Megatron reads --local-rank arg
    parser.set_defaults(local_rank=int(os.environ.get('LOCAL_RANK', 0)))
    return parser


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_megatron_model(args):
    """Create a Megatron model and load checkpoint (gathering TP/PP/EP shards).

    Returns the model with a full (un-sharded) state dict loaded.
    """
    bridge = Qwen2Bridge()

    args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
    args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
    if args.num_experts is not None:
        args.expert_model_parallel_size = args.target_expert_model_parallel_size
    if args.tensor_model_parallel_size > 1:
        args.sequence_parallel = True

    assert args.num_query_groups >= args.target_tensor_model_parallel_size

    # Copy config/tokenizer files
    for target_dir in [args.save, args.load]:
        os.makedirs(target_dir, exist_ok=True)
        for pattern in ["config*.json", "generation_config.json",
                        "tokenizer*", "vocab.json", "merges.txt"]:
            os.system(f"cp -rf {args.hf_ckpt_path}/{pattern} {target_dir} 2>/dev/null")

    model = make_model_provider("qwen2")()

    # Load and gather checkpoint shards via bridge
    state_dict = bridge.load_mg_state_dict(args)
    model.load_state_dict(state_dict, strict=False)

    return model


# ---------------------------------------------------------------------------
# HF model saving
# ---------------------------------------------------------------------------

def save_hfmodel(args, model):
    """Save an HF model in sharded safetensors or PyTorch format."""
    output_state_dict = model.state_dict()
    max_shard_size = "10GB"
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)
    os.makedirs(args.save, exist_ok=True)

    for shard_file, shard in shards.items():
        if args.save_safetensors:
            shard_file = shard_file.replace("pytorch_", "")
            shard_file = shard_file.replace(".bin", ".safetensors")
            target_file = os.path.join(args.save, shard_file)
            print(f'huggingface model is saved to {target_file}')
            new_shard = {k: copy.deepcopy(v) for k, v in shard.items()}
            safetensors.torch.save_file(
                clone_state_dict(new_shard), target_file,
                metadata={"format": "pt"},
            )
        else:
            target_file = os.path.join(args.save, shard_file)
            print(f'huggingface model is saved to {target_file}')
            torch.save(clone_state_dict(shard), target_file)

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save, WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint "
            f"({max_shard_size}) and is going to be split in {len(shards)} "
            f"checkpoint shards. You can find where each parameters has been "
            f"saved in the index located at {save_index_file}."
        )


# ---------------------------------------------------------------------------
# Forward comparison (debugging tool)
# ---------------------------------------------------------------------------

def check_hf_mg_forward(hfmodel, mgmodel, mgargs):
    """Run a forward pass on both HF and Megatron models and compare outputs.

    Used to verify conversion correctness.
    """
    hf_hiddens = [{} for _ in range(mgargs.num_layers)]
    mg_hiddens = [{} for _ in range(mgargs.num_layers)]

    hidden_size = mgargs.hidden_size
    vocab_size = mgargs.padded_vocab_size

    def print_input_hook(module, args, kwargs, layer_idx, mode):
        frame, name = mode.split('-')
        if frame == 'hf':
            hf_hiddens[layer_idx][name] = args[0].transpose(0, 1)
        elif frame == 'mg' and 'layer' in mode:
            mg_hiddens[layer_idx][name] = kwargs.get('hidden_states')
        elif frame == 'mg':
            mg_hiddens[layer_idx][name] = args[0]

    def print_output_hook(module, args, kwargs, output, layer_idx, mode):
        frame, name = mode.split('-')
        if mode in ['hf-lmhead']:
            hf_hiddens[layer_idx][name] = output.transpose(0, 1).reshape(-1, vocab_size)
            hf_hiddens[layer_idx][name + "_weight"] = module.weight
            hf_hiddens[layer_idx][name + '_token'] = output.transpose(0, 1).max(dim=-1)[1]
        elif mode in ['mg-lmhead']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, vocab_size)
            mg_hiddens[layer_idx][name + "_weight"] = module.weight
            mg_hiddens[layer_idx][name + '_token'] = output[0].max(dim=-1)[1]
        elif mode in ['hf-o_proj_out']:
            hf_hiddens[layer_idx][name] = output
            hf_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['mg-o_proj_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
            mg_hiddens[layer_idx][name + '_weight'] = module.weight
        elif mode in ['hf-attn_out']:
            hf_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)
        elif mode in ['mg-attn_out']:
            mg_hiddens[layer_idx][name] = output[0].reshape(-1, hidden_size)

    if mgargs.untie_embeddings_and_output_weights:
        hfmodel.lm_head.register_forward_hook(
            partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='hf-lmhead'),
            with_kwargs=True,
        )
        mgmodel.output_layer.register_forward_hook(
            partial(print_output_hook, layer_idx=mgargs.num_layers - 1, mode='mg-lmhead'),
            with_kwargs=True,
        )

    for idx, layer in enumerate(hfmodel.model.layers):
        layer.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hf-layer_in'), with_kwargs=True,
        )
        layer.self_attn.o_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='hf-o_proj_in'), with_kwargs=True,
        )
        layer.self_attn.o_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hf-o_proj_out'), with_kwargs=True,
        )
        layer.self_attn.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='hf-attn_out'), with_kwargs=True,
        )

    for idx, layer in enumerate(mgmodel.decoder.layers):
        layer.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-layer_in'), with_kwargs=True,
        )
        layer.self_attention.linear_proj.register_forward_pre_hook(
            partial(print_input_hook, layer_idx=idx, mode='mg-o_proj_in'), with_kwargs=True,
        )
        layer.self_attention.linear_proj.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-o_proj_out'), with_kwargs=True,
        )
        layer.self_attention.register_forward_hook(
            partial(print_output_hook, layer_idx=idx, mode='mg-attn_out'), with_kwargs=True,
        )

    input_ids = torch.tensor(
        [[151644, 8506, 22564, 27608, 75188, 4344, 121395, 61991, 79554, 36689]]
    ).long().cuda()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        input_ids, -100, True, True, True
    )
    print(hfmodel)
    print(mgmodel)

    is_oom = False
    with torch.inference_mode():
        try:
            hfmodel.cuda()
            hflogits = hfmodel(input_ids=input_ids).logits
        except torch.cuda.OutOfMemoryError:
            print('oom for huggingface model forward')
            is_oom = True
        hfmodel.cpu()
        del hfmodel

    with torch.inference_mode():
        try:
            mgmodel.cuda()
            mglogits = mgmodel(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids,
            )
        except torch.cuda.OutOfMemoryError:
            print('oom for megatron model forward')
            is_oom = True
        mgmodel.cpu()
        del mgmodel

    epsilon = 1e-5
    for idx, (hfh, mgh) in enumerate(zip(hf_hiddens, mg_hiddens)):
        assert len(hfh) == len(mgh)
        for k, hfv in hfh.items():
            mgv, hfv = mgh[k].cpu(), hfv.cpu()
            same_num = (hfv != mgv).sum()
            diff_num = ((hfv - mgv) > epsilon).sum()
            diff_max = (hfv - mgv).abs().max()
            print(
                f'layer:{idx}, {k}, diff: {same_num}, '
                f'diff>{epsilon}:[{diff_num}/{hfv.numel()}] diff_max:{diff_max}'
            )

    if not is_oom:
        same_num = (hflogits != mglogits).sum()
        diff_num = ((hflogits - mglogits) > epsilon).sum()
        diff_max = (hflogits - mglogits).abs().max()
        print(
            f'logits: {same_num}, diff>{epsilon}:'
            f'[{diff_num}/{hflogits.numel()}] diff_max:{diff_max}'
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    initialize_megatron(extra_args_provider=add_extra_args)
    args = get_args()

    # NullTokenizer sets padded_vocab_size=0; fix it from HF config.json.
    if args.padded_vocab_size == 0:
        import math
        hf_cfg = AutoConfig.from_pretrained(args.model_path)
        vocab_size = hf_cfg.vocab_size
        multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
        args.padded_vocab_size = int(math.ceil(vocab_size / multiple) * multiple)

    bridge = Qwen2Bridge()

    if args.convert_checkpoint_from_megatron_to_transformers:
        # Megatron -> HF
        config = AutoConfig.from_pretrained(args.hf_ckpt_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.hf_ckpt_path, torch_dtype=config.torch_dtype,
        )
        mg_model = load_megatron_model(args)
        bridge.mg_to_hf(mg_model, hf_model, args)
        save_hfmodel(args, hf_model)
    else:
        # HF -> Megatron
        config = AutoConfig.from_pretrained(args.load)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.load, torch_dtype=config.torch_dtype,
        )
        mg_model = make_model_provider("qwen2")()
        bridge.hf_to_mg(hf_model, mg_model, args)
        if getattr(args, 'test_convert_precision', False) and not args.num_experts:
            check_hf_mg_forward(hf_model, mg_model, args)
        bridge.save_mg_checkpoint(mg_model, args)


if __name__ == "__main__":
    main()
