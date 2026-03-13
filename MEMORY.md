# AutoAlign Megatron Refactor - 项目记忆

## 项目概述
重构 AutoAlign 的 Megatron 模块，参考 ms-swift 的设计模式，提升可维护性和可扩展性。

## 当前状态 (2026-03-10, SFT 训练已通过)

### Phase 0-2 已完成 ✅
- **P0**: 包结构迁移 (autoalign_megatron → autoalign.megatron)
- **P1**: Config 自动推导、在线数据处理、统一 CLI 入口
- **P2**: Bridge 抽象层、模型注册机制、去 Fork 训练循环

### 权重转换 已完成 ✅
- HF→Megatron / Megatron→HF 双向转换，round-trip max_diff=0.0

### SFT 训练 已通过 ✅ (2026-03-10)
- **NPU 脚本**: `scripts/train/megatron/train/npu/sft_conv.sh`
- **配置**: TP=2 PP=2, DEVICES=8,9,10,11, Qwen2.5-7B, BATCH_SIZE=1, GBS=4
- **结果**: 5 iterations, loss 16.37→13.14, throughput ~114 TFLOP/s/GPU
- **内存**: ~32.7GB/41.2GB (PP stage 0), ~32.8GB/36GB (PP stage 1)

### SFT 训练修复的 Bug
1. **model_config.py**: `vocab_size = hf_vocab_size - 1` — NullTokenizer 加 +1 eod，需要补偿
2. **arguments.py**: `--model-type` → `--model-type-name` — 避免 Megatron ModelType enum 覆盖
3. **registry.py**: try/except `core_transformer_config_from_args` — MindSpeed 1-arg wrapper
4. **sft.py**: patch `load_checkpoint` 为 strict=False — TE _extra_state 在 NPU 上缺失
5. **utils.py**: `get_ltor_masks_and_position_ids` 改为 5 参数 — 新 Megatron 移除了 create_attention_mask_in_dataloader 参数
6. **utils.py**: `cur_max_seq_length` broadcast 形状统一为 `(1,)` — 0-dim vs 1-dim tensor 不匹配
7. **utils.py**: attention_mask broadcast 不对称修复 — MindSpeed 设 create_attention_mask_in_dataloader=False 时，rank 0 仍然 broadcast 但 rank 1 跳过，导致后续 broadcast 全部错位
8. **scripts**: NVTE_* env vars → `--attention-backend flash` + `--use-flash-attn` — Megatron-Core ≥0.12 + MindSpeed 各需不同参数
9. **npu scripts**: 移除 `--gradient-accumulation-fusion false` 和 `--log-batch-size-to-tensorboard` — MindSpeed 不支持

### 服务器环境
- 地址: root@115.120.90.192:8022, SSH key: id_rsa_910c
- 容器: `funny_sammet`, 代码: `/home/ma-user/AutoAlign`
- 分支: megatron-refactor (最新 commit: b0cae04)
- GitHub 无法从服务器访问，需通过 scp+docker cp 同步文件
- NPU 0-3 被占用，使用 chips 8-15 (ASCEND_RT_VISIBLE_DEVICES=8,9,10,11)

### 关键 MindSpeed 行为（踩坑记录）
- `args.transformer_impl = 'transformer_engine'` → 强制改为 `'local'`
- `args.create_attention_mask_in_dataloader = False` 当 use_flash_attn=True
- `core_transformer_config_from_args` 被 wrap 为 1-arg 函数
- `torch.cuda.*` → `torch.npu.*` 通过 transfer_to_npu 猴补丁
- HCCL 占用 HCCL_IF_BASE_PORT 开始的 16 个连续端口

### 下一步行动
1. 运行 DPO 训练脚本验证
2. 开始 Phase 3 开发 (LoRA, Packing, 更多模型)

## 关键设计模式

### Bridge Pattern
- `GPTBridge` 基类封装 HF↔Megatron 双向转换
- `Qwen2Bridge` 子类处理 QKV merge/split、MoE expert 等

### Registry Pattern
- `MegatronModelMeta` + `register_megatron_model()` + `make_model_provider()`

### Config Auto-Derivation
- `derive_megatron_args()` 从 HF config.json 自动推导参数
- `inject_hf_model_args()` 在 argparse 时注入，CLI 参数优先

### Online Tokenization
- `OnlineSFTDataset` / `OnlineDPODataset` 运行时 tokenize
- 复用 AutoAlign 的 Template 系统
