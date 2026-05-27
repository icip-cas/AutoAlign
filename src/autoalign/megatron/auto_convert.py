"""Auto HF↔Megatron checkpoint conversion for ``autoalign-cli megatron-sft``.

Orchestrates the existing single-rank conversion entry
(``autoalign.megatron.toolkits.checkpoint.qwen.common``) before and after
training so users can pass ``--model_name_or_path`` (HF dir) and get HF output
back without touching mcore explicitly.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import random
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

CONVERT_MODULE = "autoalign.megatron.toolkits.checkpoint.qwen.common"
CACHE_COMPLETE_MARKER = ".complete"
PRE_CONVERT_COMPLETE_MARKER = ".pre_convert_complete"
POST_CONVERT_COMPLETE_MARKER = ".post_convert_complete"
SUPPORTED_MODEL_TYPES = {"qwen2", "qwen3", "qwen2_moe", "qwen3_moe"}


@dataclass
class ConvertSpec:
    hf_path: Path
    mcore_input: Path
    output_dir: Path
    tp: int
    pp: int
    ep: int
    dtype: str | None  # "bf16" / "fp16" / None (fp32)
    transformer_impl: str
    skip_pre: bool
    skip_post: bool
    reuse_cache: bool  # True only when AUTOALIGN_MCORE_CACHE is set


def _find_arg(argv: Sequence[str], name: str, default: str | None = None) -> str | None:
    argv = list(argv)
    try:
        i = argv.index(name)
    except ValueError:
        return default
    if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
        return default
    return argv[i + 1]


def _has_flag(argv: Sequence[str], name: str) -> bool:
    return name in argv


def _detect_model_type(hf_path: Path) -> str | None:
    config = hf_path / "config.json"
    if not config.exists():
        return None
    with config.open() as f:
        return json.load(f).get("model_type")


def _cache_key(
    hf_path: Path, tp: int, pp: int, ep: int, dtype: str | None, transformer_impl: str
) -> str:
    key = f"{hf_path}|{tp}|{pp}|{ep}|{dtype or 'fp32'}|{transformer_impl}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _resolve_mcore_input_dir(
    output_dir: Path,
    hf_path: Path,
    tp: int,
    pp: int,
    ep: int,
    dtype: str | None,
    transformer_impl: str,
) -> tuple[Path, bool]:
    """Pick where mcore input should live.

    Default is ephemeral: ``<output_dir>/.mcore_input`` — overwritten each run.
    If ``AUTOALIGN_MCORE_CACHE_DIR`` is set, opt into hash-keyed cross-run reuse
    at ``$AUTOALIGN_MCORE_CACHE_DIR/<hf_name>-<hash>``.
    """
    env = os.environ.get("AUTOALIGN_MCORE_CACHE_DIR")
    if env:
        cache_hash = _cache_key(hf_path, tp, pp, ep, dtype, transformer_impl)
        return Path(env) / f"{hf_path.name}-{cache_hash}", True
    return output_dir / ".mcore_input", False


def _is_cache_valid(cache_dir: Path, hf_path: Path) -> bool:
    marker = cache_dir / CACHE_COMPLETE_MARKER
    if not marker.exists():
        return False
    config = hf_path / "config.json"
    if config.exists() and config.stat().st_mtime > marker.stat().st_mtime:
        return False
    return True


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _pre_convert_metadata(spec: ConvertSpec) -> dict:
    return {
        "hf_path": str(spec.hf_path),
        "tp": spec.tp,
        "pp": spec.pp,
        "ep": spec.ep,
        "dtype": spec.dtype,
        "transformer_impl": spec.transformer_impl,
    }


def _is_pre_convert_ready(spec: ConvertSpec) -> bool:
    marker = spec.mcore_input / PRE_CONVERT_COMPLETE_MARKER
    if not marker.exists():
        return False
    try:
        with marker.open() as f:
            if json.load(f) != _pre_convert_metadata(spec):
                return False
    except Exception:
        return False
    config = spec.hf_path / "config.json"
    if config.exists() and config.stat().st_mtime > marker.stat().st_mtime:
        return False
    return True


def _write_pre_convert_marker(spec: ConvertSpec) -> None:
    marker = spec.mcore_input / PRE_CONVERT_COMPLETE_MARKER
    with marker.open("w") as f:
        json.dump(_pre_convert_metadata(spec), f, sort_keys=True)


def _is_post_convert_ready(spec: ConvertSpec) -> bool:
    marker = spec.output_dir / POST_CONVERT_COMPLETE_MARKER
    if not marker.exists():
        return False
    mcore_marker = spec.output_dir / "mcore" / "latest_checkpointed_iteration.txt"
    if mcore_marker.exists() and mcore_marker.stat().st_mtime > marker.stat().st_mtime:
        return False
    return True


def _get_node_rank() -> int:
    for name in ("NODE_RANK", "RANK"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 0


def _wait_for_pre_convert(spec: ConvertSpec) -> None:
    lock_path = spec.mcore_input.parent / f".{spec.mcore_input.name}.lock"
    timeout = int(os.environ.get("AUTOALIGN_CONVERT_WAIT_TIMEOUT", "86400"))
    deadline = time.monotonic() + timeout
    logger.info(f"[auto-convert] waiting for rank-0 pre-convert: {spec.mcore_input}")

    while True:
        with _file_lock(lock_path):
            if _is_pre_convert_ready(spec):
                logger.info(f"[auto-convert] mcore input ready at: {spec.mcore_input}")
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for pre-convert output: {spec.mcore_input}")
        time.sleep(5)


def build_convert_spec(
    translated_argv: Sequence[str],
    options: dict,
    world_size: int,
) -> ConvertSpec | None:
    hf_path_str = _find_arg(translated_argv, "--model-path")
    output_dir_str = _find_arg(translated_argv, "--save")
    user_passed_load = _has_flag(translated_argv, "--load")
    no_export_hf = options.get("no_export_hf", False)

    if user_passed_load and no_export_hf:
        return None
    if not hf_path_str:
        return None
    if not output_dir_str:
        raise ValueError("--output_dir / --save is required to anchor auto-convert outputs.")

    hf_path = Path(hf_path_str).resolve()
    output_dir = Path(output_dir_str).resolve()

    model_type = _detect_model_type(hf_path)
    if model_type is None:
        raise ValueError(f"Could not detect model_type from {hf_path}/config.json")
    if not user_passed_load and model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Auto-convert not supported for model_type={model_type!r}. "
            "Pass --load <mcore_path> explicitly (and optionally --no-export-hf)."
        )

    tp = int(_find_arg(translated_argv, "--tensor-model-parallel-size", "1"))
    pp = int(_find_arg(translated_argv, "--pipeline-model-parallel-size", "1"))
    ep = int(_find_arg(translated_argv, "--expert-model-parallel-size", "1"))
    if _has_flag(translated_argv, "--bf16"):
        dtype: str | None = "bf16"
    elif _has_flag(translated_argv, "--fp16"):
        dtype = "fp16"
    else:
        dtype = None
    transformer_impl = _find_arg(translated_argv, "--transformer-impl", "local") or "local"

    mcore_input, reuse_cache = _resolve_mcore_input_dir(
        output_dir, hf_path, tp, pp, ep, dtype, transformer_impl
    )

    return ConvertSpec(
        hf_path=hf_path,
        mcore_input=mcore_input,
        output_dir=output_dir,
        tp=tp,
        pp=pp,
        ep=ep,
        dtype=dtype,
        transformer_impl=transformer_impl,
        skip_pre=user_passed_load,
        skip_post=no_export_hf or user_passed_load,
        reuse_cache=reuse_cache,
    )


def redirect_save_to_mcore(translated_argv: Sequence[str]) -> list[str]:
    """Rewrite ``--save X`` to ``--save X/mcore``.

    The HF export lands at the top-level ``--save`` dir so downstream tools
    can point at ``--output_dir`` and find a normal HF model layout.
    """
    out = list(translated_argv)
    try:
        i = out.index("--save")
    except ValueError:
        return out
    if i + 1 < len(out) and not out[i + 1].startswith("--"):
        out[i + 1] = str(Path(out[i + 1]) / "mcore")
    return out


def _torchrun_prefix() -> list[str]:
    master_port = str(random.randint(20001, 29999))
    return [
        "torchrun",
        "--nnodes", "1",
        "--node_rank", "0",
        "--nproc_per_node", "1",
        "--master_addr", "127.0.0.1",
        "--master_port", master_port,
    ]


def _common_convert_args(spec: ConvertSpec) -> list[str]:
    args = [
        "--model-path", str(spec.hf_path),
        "--target-tensor-model-parallel-size", str(spec.tp),
        "--target-pipeline-model-parallel-size", str(spec.pp),
        "--target-expert-model-parallel-size", str(spec.ep),
        "--transformer-impl", spec.transformer_impl,
        "--micro-batch-size", "1",
        "--save-interval", "1",
        "--seq-length", "1",
        "--no-async-tensor-model-parallel-allreduce",
        "--no-bias-swiglu-fusion",
        "--no-rope-fusion",
        "--use-mcore-models",
        "--use-cpu-initialization",
    ]
    if spec.dtype in {"bf16", "fp16"}:
        args.append(f"--{spec.dtype}")
    return args


def _hf_to_mcore_cmd(spec: ConvertSpec) -> list[str]:
    args = _common_convert_args(spec) + [
        "--load", str(spec.hf_path),
        "--save", str(spec.mcore_input),
    ]
    return [*_torchrun_prefix(), "-m", CONVERT_MODULE, *args]


def _mcore_to_hf_cmd(spec: ConvertSpec) -> list[str]:
    args = _common_convert_args(spec) + [
        "--convert-checkpoint-from-megatron-to-transformers",
        "--hf-ckpt-path", str(spec.hf_path),
        "--load", str(spec.output_dir / "mcore"),
        "--save", str(spec.output_dir),
        "--save-safetensors",
    ]
    return [*_torchrun_prefix(), "-m", CONVERT_MODULE, *args]


def run_pre_convert(spec: ConvertSpec, dry_run: bool) -> None:
    cmd = _hf_to_mcore_cmd(spec)

    if spec.reuse_cache and _is_cache_valid(spec.mcore_input, spec.hf_path):
        logger.info(f"[auto-convert] mcore cache hit: {spec.mcore_input}")
        if dry_run:
            logger.info("[auto-convert] dry-run: pre-convert would be skipped")
        return

    if not spec.reuse_cache and _get_node_rank() != 0:
        if dry_run:
            logger.info("[auto-convert] dry-run: non-zero node would wait for rank-0 pre-convert")
            return
        _wait_for_pre_convert(spec)
        return

    if spec.reuse_cache and _is_pre_convert_ready(spec):
        logger.info(f"[auto-convert] mcore input already ready: {spec.mcore_input}")
        return

    logger.info(f"[auto-convert] pre-convert HF→mcore → {spec.mcore_input}")
    logger.info(f"[auto-convert] pre-convert command: {shlex.join(cmd)}")
    if dry_run:
        return

    lock_path = spec.mcore_input.parent / f".{spec.mcore_input.name}.lock"
    logger.info(f"[auto-convert] waiting for pre-convert lock: {lock_path}")
    with _file_lock(lock_path):
        if spec.reuse_cache:
            if _is_cache_valid(spec.mcore_input, spec.hf_path):
                logger.info(f"[auto-convert] mcore cache hit after lock: {spec.mcore_input}")
                return
            if _is_pre_convert_ready(spec):
                logger.info(f"[auto-convert] mcore input became ready: {spec.mcore_input}")
                return

        if spec.mcore_input.exists():
            shutil.rmtree(spec.mcore_input)
        spec.mcore_input.mkdir(parents=True)

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"HF→mcore conversion failed (exit {result.returncode}); "
                f"partial output at {spec.mcore_input}"
            )

        _write_pre_convert_marker(spec)
        if spec.reuse_cache:
            (spec.mcore_input / CACHE_COMPLETE_MARKER).touch()
    logger.info(f"[auto-convert] mcore input ready at: {spec.mcore_input}")


def run_post_convert(spec: ConvertSpec, dry_run: bool) -> None:
    cmd = _mcore_to_hf_cmd(spec)
    logger.info(f"[auto-convert] post-convert command: {shlex.join(cmd)}")
    if dry_run:
        return
    if _get_node_rank() != 0:
        logger.info("[auto-convert] skipping post-convert on non-zero node")
        return

    mcore_dir = spec.output_dir / "mcore"
    marker = mcore_dir / "latest_checkpointed_iteration.txt"
    if not marker.exists():
        raise RuntimeError(
            f"No mcore checkpoint found at {mcore_dir}; did training actually save?"
        )

    lock_path = spec.output_dir / ".post_convert.lock"
    logger.info(f"[auto-convert] waiting for post-convert lock: {lock_path}")
    with _file_lock(lock_path):
        if _is_post_convert_ready(spec):
            logger.info(f"[auto-convert] HF export already ready: {spec.output_dir}")
            return

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"mcore→HF conversion failed (exit {result.returncode})")

        (spec.output_dir / POST_CONVERT_COMPLETE_MARKER).touch()
    logger.info(f"[auto-convert] HF export written to: {spec.output_dir}")
