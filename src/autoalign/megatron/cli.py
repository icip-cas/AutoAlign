"""HF-style CLI translator for Megatron entries.

Users invoke ``autoalign-cli megatron-sft`` with HF-style flags
(``--model_name_or_path``, ``--per_device_train_batch_size``, ``--bf16 True``,
…). This module rewrites that argv into Megatron kebab-case flags that
``autoalign.megatron.entries.sft`` understands.
"""

from typing import Sequence


# Flags consumed by the adaptor itself (not forwarded to Megatron).
LOCAL_CONSUMED_FLAGS = {
    "--no-export-hf",  # skip post-training mcore→HF conversion
}


TRANSLATION_TABLE = {
    "--model_name_or_path": "--model-path",
    "--data_path": "--data-path",
    "--conv_template_name": "--template",
    "--output_dir": "--save",
    "--num_train_epochs": "--epochs",
    "--per_device_train_batch_size": "--micro-batch-size",
    "--learning_rate": "--lr",
    "--weight_decay": "--weight-decay",
    "--warmup_ratio": "--lr-warmup-fraction",
    "--lr_scheduler_type": "--lr-decay-style",
    "--logging_steps": "--log-interval",
    "--save_steps": "--save-interval",
    "--eval_steps": "--eval-interval",
    "--max_grad_norm": "--clip-grad",
    "--report_to": "--report-to",
    "--model_max_length": "--seq-length",
    "--cutoff_len": "--seq-length",
    "--bf16": "--bf16",
    "--fp16": "--fp16",
}

PARALLEL_ALIAS_TABLE = {
    "--tp": "--tensor-model-parallel-size",
    "--pp": "--pipeline-model-parallel-size",
    "--cp": "--context-parallel-size",
    "--ep": "--expert-model-parallel-size",
    "--sp": "--sequence-parallel",
}

REJECTED_FLAGS = {
    "--deepspeed": "Megatron uses its own distributed optimizer; remove --deepspeed.",
    "--enable_liger_kernel": "Liger kernel is HF-only; Megatron uses TransformerEngine.",
    "--lazy_preprocess": "Megatron's --dataset json is always online; remove --lazy_preprocess.",
    "--neat_packing": "Packing is not implemented for Megatron yet; remove --neat_packing.",
    "--packing_strategy": "Packing is not implemented for Megatron yet; remove --packing_strategy.",
    "--gradient_checkpointing": "Use --recompute-granularity full --recompute-method uniform instead of --gradient_checkpointing.",
    "--per_device_eval_batch_size": "Megatron reuses global batch semantics for eval; remove --per_device_eval_batch_size.",
    "--logging_dir": "Set SWANLAB_LOG_DIR via environment variable instead of --logging_dir.",
    "--ddp_timeout": "Use --distributed-timeout-minutes (minutes, not seconds) instead of --ddp_timeout.",
    "--ignore_pad_token_for_loss": "Megatron handles loss masking via its own loss_mask path; remove --ignore_pad_token_for_loss.",
}

# HF-specific flags with no Megatron analog that aren't worth surfacing as
# errors — quietly consume so users can copy whole shell blocks unchanged.
SILENTLY_DROPPED_HF_FLAGS = {
    "--save_strategy",
    "--eval_strategy",
    "--evaluation_strategy",
    "--save_total_limit",
    "--load_best_model_at_end",
    "--metric_for_best_model",
    "--greater_is_better",
    "--overwrite_output_dir",
    "--do_train",
    "--do_eval",
    "--prediction_loss_only",
    "--dataloader_num_workers",
    "--dataloader_pin_memory",
    "--remove_unused_columns",
    "--label_names",
}

# Flags emitted by the translator that take ``store_true`` semantics (no value).
BOOLEAN_OUTPUT_FLAGS = {
    "--bf16",
    "--fp16",
    "--sequence-parallel",
    "--variable-seq-lengths",
}


class TranslationError(ValueError):
    pass


def _looks_like_flag(token: str) -> bool:
    return token.startswith("--")


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return True
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise TranslationError(f"Invalid boolean value: {value}")


def _snake_to_kebab(flag: str) -> str:
    if not flag.startswith("--"):
        return flag
    return "--" + flag[2:].replace("_", "-")


def _take_value(argv: Sequence[str], i: int, flag: str) -> str:
    if i + 1 >= len(argv) or _looks_like_flag(argv[i + 1]):
        raise TranslationError(f"Missing value for {flag}")
    return argv[i + 1]


def _peek_value(argv: Sequence[str], i: int) -> str | None:
    if i + 1 < len(argv) and not _looks_like_flag(argv[i + 1]):
        return argv[i + 1]
    return None


def _get_arg_value(argv: Sequence[str], names: Sequence[str], default: int = 1) -> int:
    for name in names:
        if name in argv:
            index = argv.index(name)
            if index + 1 >= len(argv) or _looks_like_flag(argv[index + 1]):
                raise TranslationError(f"Missing value for {name}")
            return int(argv[index + 1])
    return default


def _get_str_arg_value(argv: Sequence[str], names: Sequence[str], default: str | None = None) -> str | None:
    for name in names:
        if name in argv:
            index = argv.index(name)
            if index + 1 >= len(argv) or _looks_like_flag(argv[index + 1]):
                raise TranslationError(f"Missing value for {name}")
            return argv[index + 1]
    return default


def _ensure_no_alias_conflict(argv: Sequence[str]) -> None:
    for alias, target in PARALLEL_ALIAS_TABLE.items():
        if alias in argv and target in argv:
            raise TranslationError(
                f"Conflicting flags: pass either {alias} or {target}, not both."
            )


def _compute_global_batch_size(
    translated: Sequence[str],
    grad_accumulation: int,
    world_size: int,
) -> str | None:
    if "--global-batch-size" in translated:
        return None
    if "--micro-batch-size" not in translated:
        return None

    micro_batch_size = _get_arg_value(translated, ["--micro-batch-size"])
    tp = _get_arg_value(translated, ["--tensor-model-parallel-size"])
    pp = _get_arg_value(translated, ["--pipeline-model-parallel-size"])
    cp = _get_arg_value(translated, ["--context-parallel-size"])

    parallel_product = tp * pp * cp
    if parallel_product <= 0:
        raise TranslationError("Parallel sizes must be positive integers.")
    if world_size % parallel_product != 0:
        raise TranslationError(
            f"world_size={world_size} is not divisible by tp*pp*cp={parallel_product}; "
            "cannot derive --global-batch-size."
        )

    data_parallel_size = world_size // parallel_product
    return str(micro_batch_size * grad_accumulation * data_parallel_size)


def translate(argv: Sequence[str], world_size: int) -> tuple[list[str], bool, dict]:
    """Translate HF-style argv to Megatron kebab-case argv.

    Returns ``(translated_argv, dry_run, options)``. ``options`` captures
    adaptor-consumed flags (e.g. ``--no-export-hf``) the orchestrator needs.
    """
    argv = list(argv)
    _ensure_no_alias_conflict(argv)

    dry_run = False
    options: dict = {"no_export_hf": False}
    grad_accumulation = 1
    variable_seq_lengths_specified = any(
        _snake_to_kebab(token) == "--variable-seq-lengths" for token in argv
    )
    translated: list[str] = []
    i = 0

    while i < len(argv):
        arg = argv[i]

        if arg == "--dry-run":
            dry_run = True
            i += 1
            continue

        if arg == "--no-export-hf":
            options["no_export_hf"] = True
            i += 1
            continue

        if arg in REJECTED_FLAGS:
            raise TranslationError(REJECTED_FLAGS[arg])

        if arg in SILENTLY_DROPPED_HF_FLAGS:
            i += 2 if _peek_value(argv, i) is not None else 1
            continue

        if arg == "--gradient_accumulation_steps":
            grad_accumulation = int(_take_value(argv, i, arg))
            i += 2
            continue

        if arg in PARALLEL_ALIAS_TABLE:
            target = PARALLEL_ALIAS_TABLE[arg]
            if target in BOOLEAN_OUTPUT_FLAGS:
                next_value = _peek_value(argv, i)
                if _parse_bool(next_value):
                    translated.append(target)
                i += 2 if next_value is not None else 1
                continue
            translated.extend([target, _take_value(argv, i, arg)])
            i += 2
            continue

        mapped = TRANSLATION_TABLE.get(arg)
        if mapped is not None:
            if mapped in BOOLEAN_OUTPUT_FLAGS:
                next_value = _peek_value(argv, i)
                if _parse_bool(next_value):
                    translated.append(mapped)
                i += 2 if next_value is not None else 1
                continue
            translated.extend([mapped, _take_value(argv, i, arg)])
            i += 2
            continue

        # Pass-through: native kebab-case Megatron flags (e.g. --load,
        # --use-distributed-optimizer) or any unknown flag. Snake-case forms get
        # converted as a last-resort fallback.
        kebab_arg = _snake_to_kebab(arg)
        if kebab_arg in BOOLEAN_OUTPUT_FLAGS:
            next_value = _peek_value(argv, i)
            if _parse_bool(next_value):
                translated.append(kebab_arg)
            i += 2 if next_value is not None else 1
            continue

        translated.append(kebab_arg)
        next_value = _peek_value(argv, i)
        if next_value is not None:
            translated.append(next_value)
            i += 2
        else:
            i += 1

    if "--dataset" not in translated:
        translated.extend(["--dataset", "json"])
    if "--split" not in translated:
        translated.extend(["--split", "100,0,0"])

    global_batch_size = _compute_global_batch_size(
        translated, grad_accumulation=grad_accumulation, world_size=world_size
    )
    if global_batch_size is not None:
        translated.extend(["--global-batch-size", global_batch_size])

    dataset = _get_str_arg_value(translated, ["--dataset"], default="json")
    pp = _get_arg_value(translated, ["--pipeline-model-parallel-size"])
    if (
        dataset == "json"
        and pp > 1
        and not variable_seq_lengths_specified
        and "--variable-seq-lengths" not in translated
    ):
        translated.append("--variable-seq-lengths")

    return translated, dry_run, options
