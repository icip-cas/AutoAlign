# Review Response

## `src/autoalign/megatron/patch/model/model_dpo.py`

**Reviewer comment:** This file is a near-duplicate of `src/autoalign/megatron/patch/model/qwen2/model_dpo.py`, but contains an incorrect `get_batch_logps` implementation that does not handle `-100` ignored labels correctly.

**Response:** 已删除。这个文件是旧的重复实现，当前 DPO 入口使用的是 `src/autoalign/megatron/patch/model/qwen2/model_dpo.py`，删除后不会影响现有引用。同时删除了旧命名空间 `src/autoalign_megatron/`，避免继续保留重复且过期的 DPO 实现。

## `docker/Dockerfile.megatron-npu-910b`

**Reviewer comment:** merge 之后还要依赖 `megatron-refactor` branch 的环境吗？

**Response:** 已处理。默认 `AUTOALIGN_BRANCH` 已从 `megatron-refactor` 改为 `main`，merge 后 Docker 构建不再默认依赖临时开发分支。同步检查并修正了其它 Megatron Dockerfile 中相同的默认分支配置。

## `scripts/train/megatron/convert/qwen2_5/convert_hf_to_mcore.sh`

**Reviewer comment:** 统一不开 TE 了吗？

**Response:** 已改回。这个 qwen2_5 转换脚本服务 GPU/TE 训练路径，应与训练脚本默认的 `--transformer-impl transformer_engine` 保持一致；默认 `USE_TE` 已恢复为 `true`，默认输出目录也从 `*-local-*` 改回 `*-te-*`。同步修正了 qwen2_5 的反向转换脚本 `convert_mcore_to_hf.sh` 和 DPO 反向转换脚本 `convert_mcore_to_hf_dpo.sh`，避免同类默认值不一致。NPU 专用转换脚本仍保持 local 默认。
