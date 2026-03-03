# AutoAlign Megatron Refactor

## Background

对比了 AutoAlign 和 [ms-swift](https://github.com/modelscope/ms-swift) 的 Megatron 接入方式（重点参考 [Mcore-Bridge 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Megatron-SWIFT/Mcore-Bridge.md)）。ms-swift 的设计在可维护性、可扩展性和用户体验上显著优于当前实现。本分支目标是重构 AutoAlign 的 Megatron 模块。

## 当前实现的核心问题

1. **包结构割裂**：`autoalign_megatron` 是独立于 `autoalign` 的顶级包，无法复用 autoalign 已有的 Template、DataCollator 等基础设施，且 `pyproject.toml` 中 `autoalign_megatron = []` 依赖为空壳
2. **环境安装脆弱**：`env_install.sh` 将 Megatron-LM 和 Pai-Megatron-Patch 源码直接复制到 site-packages，不走正规包管理，版本锁死在特定 commit
3. **Fork 式训练循环**：`training_sft.py` / `training_dpo.py` 各 ~70KB，从 Megatron-LM 复制修改，与上游同步困难
4. **硬编码配置**：模型参数写死在 shell 脚本中
5. **离线预处理**：需要独立步骤生成 MMapIndexedDataset
6. **仅支持 Qwen2.5**，不支持 LoRA

## ms-swift 值得借鉴的设计

- **Bridge Pattern**：`GPTBridge` 基类封装 TP/PP/EP 通用逻辑，子类只需声明 `hf_state_dict_mapping`（见 `swift/megatron/model/gpt_bridge.py`）
- **Registration Pattern**：`register_megatron_model()` 一行注册一个模型家族（见 `swift/megatron/model/register.py`）
- **Config Translation**：`config_mapping` 字典自动从 HF config 推导 Megatron config（见 `swift/megatron/model/model_config.py`）
- **Monkey-Patch**：`init_megatron_env()` 运行时注入补丁而非 Fork 上游（见 `swift/megatron/init.py`）
- **轻量训练器**：委托 `get_forward_backward_func()` 等 Megatron-Core 原生 API，只自定义 `forward_step` + `loss_func`（见 `swift/megatron/trainers/`）
- **在线转换**：`SafetensorLazyLoader` + `StreamingSafetensorSaver` 训练前/后自动执行权重转换

---

## TODO: Refactor Roadmap

### Phase 0: 包结构迁移 + 依赖治理 (最高优先级)

- [x] **P0-1: 将 `autoalign_megatron` 移入 `autoalign.megatron`**
  - 已将 `src/autoalign_megatron/` 整体移动到 `src/autoalign/megatron/`
  - 已更新所有内部 import：`autoalign_megatron.xxx` → `autoalign.megatron.xxx`
  - 已更新所有 Shell 脚本中的模块路径引用
  - 已删除旧的 `src/autoalign_megatron/` 目录
  - 已在所有 6 个入口文件顶部添加 `import autoalign.megatron` 以确保 `MEGATRON_LM_PATH` 在 megatron import 之前生效

- [x] **P0-2: 依赖管理改为文档驱动（参考 ms-swift）**
  - 删除 `env_install.sh`，不在 pyproject.toml 中声明 megatron 相关依赖
  - 安装说明写入 `docs/megatron.md`，列出 pip install 命令
  - `autoalign/megatron/__init__.py` 提供运行时检查（`is_megatron_available()` 等）

- [x] **P0-3: 去掉 cp 到 site-packages 的方式**
  - 已删除 env_install.sh
  - Megatron-LM 改为 `pip install git+...`
  - Pai-Megatron-Patch 通过 PYTHONPATH 引入（无 setup.py）

### Phase 1: Quick Wins

- [x] **P1-1: Config 自动推导** — 新增 `autoalign/megatron/model_config.py`，从 HF config.json 自动提取模型参数。Shell 脚本用 `--model-path` 替代 MODEL_SIZE 硬编码块
- [x] **P1-2: 消除离线数据预处理** — 新增 `--dataset json` 模式，复用 autoalign 的 Template 系统在线 tokenize，无需离线 preprocess 步骤
- [x] **P1-3: 统一 CLI 入口** — `autoalign-cli megatron-sft/megatron-dpo` 命令，通过 `torchrun -m` 启动 Megatron 训练

### Phase 2: Architecture Refactor

- [x] **P2-1: 实现 Bridge 抽象层** — 新增 `autoalign/megatron/bridge.py`，GPTBridge 基类 + Qwen2Bridge 子类，封装 HF↔Megatron 双向权重转换（QKV merge/split、gate+up merge/split、MoE expert/shared expert）和 TP/PP/EP checkpoint 分片/聚合。`toolkits/checkpoint/qwen/common.py` 重写为使用 Bridge 的精简版（~250行 vs 原 950行），`dpo.py` 消除重复改为 thin wrapper
- [x] **P2-2: 模型注册机制** — 新增 `autoalign/megatron/registry.py`，`MegatronModelMeta` dataclass + `register_megatron_model()` + `make_model_provider()` 工厂函数。内置 Qwen2 懒注册。`sft_qwen.py` 和 `dpo_qwen.py` 的 model_provider 从 ~40行 简化为 1行调用，`common.py` 也改用 registry
- [x] **P2-3: 去 Fork 训练循环** — 将 `training_sft.py` 参数化为 SFT/DPO 共享基座（新增 `idx_file_fn`、`split_fn`、`micro_batch_multiplier` 参数），`training_dpo.py` 从 1807 行重写为 66 行 thin wrapper，消除了两个文件间 99.8% 的代码重复

### Phase 3: Feature Expansion

- [ ] **P3-1: TP-aware LoRA** — 参考 `swift/megatron/tuners/lora.py`
- [ ] **P3-2: Packing 训练** — padding-free with FlashAttention varlen
- [ ] **P3-3: 更多模型** — 基于 Bridge + Registration 快速接入 LLaMA/DeepSeek/GLM
- [ ] **P3-4: 更多训练方法** — KTO, GRPO 等
- [ ] **P3-5: 转换精度测试** — `--test_convert_precision`

---

## Key Files Reference

### AutoAlign (当前)
```
src/autoalign/megatron/
  entries/sft.py                 # 模型无关 SFT 入口 (model_type 从 --model-path 自动推导)
  entries/dpo.py                 # 模型无关 DPO 入口
  bridge.py                      # GPTBridge + Qwen2Bridge (HF↔Megatron 权重转换)
  registry.py                    # MegatronModelMeta + make_model_provider() 工厂
  model_config.py                # HF config.json → Megatron CLI args 自动推导
  patch/training_sft.py          # 共享训练循环 (SFT/DPO 参数化)
  patch/training_dpo.py          # DPO thin wrapper (66 行)
  patch/arguments.py             # 自定义参数 (--model-path, --model-type, ...)
  patch/data/                    # Dataset & batch utils
  patch/model/qwen2/             # DPO 模型
  patch/core/pipeline_parallel/  # PP 调度
  toolkits/checkpoint/qwen/      # 权重转换 (使用 Bridge)
  toolkits/sft/preprocess.py     # SFT 数据预处理
  toolkits/dpo/preprocess.py     # DPO 数据预处理
scripts/train/megatron/          # Shell 训练脚本
  convert/                       # 权重转换脚本
  preprocess/                    # 数据预处理脚本
  train/                         # 训练执行脚本
```

### ms-swift (参考)
```
swift/megatron/
  init.py                         # Monkey-patch 入口
  model/gpt_bridge.py             # Bridge 基类
  model/register.py               # 模型注册
  model/model_config.py           # Config 翻译
  trainers/base.py                # BaseMegatronTrainer
  trainers/trainer.py             # SFT Trainer
  tuners/lora.py                  # TP-aware LoRA
  convert.py                      # 转换命令
```
