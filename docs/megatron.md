# Megatron Training Pipeline

This repository is based on [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch.git) for secondary development, thanks to the original author's work. This project is used to support Alignment such as SFT and DPO etc. Currently, the operation flow of SFT and DPO is as follows.

## Environment Setup

### Prerequisites
- Python >= 3.10
- NVIDIA GPU with CUDA support (CUDA 12.x recommended)
- conda environment

### Install Dependencies

Install the Megatron training dependencies in order:

```shell
# 1. pybind11 (build dependency)
pip install pybind11

# 2. Transformer Engine
# If an installation error occurs, try adding --no-build-isolation
pip install --no-build-isolation transformer_engine[pytorch]

# 3. NVIDIA Apex (optional, for gradient_accumulation_fusion)
# Megatron can run without apex by setting --gradient-accumulation-fusion false
git clone https://github.com/NVIDIA/apex
cd apex
git checkout ac8214ee6ba77c0037c693828e39d83654d25720
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd ..

# 4. Megatron-LM
git clone https://github.com/jerryli1981/PAI-Megatron-LM-240718.git Megatron-LM
cd Megatron-LM && git checkout 7765c381d5af76f97834934a67b1e43afece02ad && cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)/Megatron-LM
export MEGATRON_LM_PATH=$(pwd)/Megatron-LM

# 5. Pai-Megatron-Patch (no setup.py, use PYTHONPATH)
git clone https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch && git checkout 9b88cc46653e2c4f7f99529f86f8737ac1da9e9a && cd ..
export PYTHONPATH=$PYTHONPATH:$(pwd)/Pai-Megatron-Patch

# 6. Flash Attention
pip install flash-attn==2.4.2 --no-build-isolation

# 7. AutoAlign itself
pip install -e .
```

> **NPU Support**: To use Ascend NPU, replace step 4 with the NPU-compatible
> Megatron-LM (e.g. MindSpeed) and point `MEGATRON_LM_PATH` to it:
> ```shell
> git clone <npu-megatron-repo> Megatron-LM-NPU
> export PYTHONPATH=$PYTHONPATH:$(pwd)/Megatron-LM-NPU
> export MEGATRON_LM_PATH=$(pwd)/Megatron-LM-NPU
> ```
> AutoAlign will automatically pick up the Megatron implementation from
> `MEGATRON_LM_PATH` at import time.

Recommended versions:

|                    | Range         | Recommended |
|--------------------|---------------|-------------|
| python             | >=3.10        | 3.10        |
| cuda               |               | 12.1        |
| torch              | >=2.5.1       | 2.5.1       |
| transformer_engine |               | 7a7225c4    |
| apex               |               | ac8214ee    |
| megatron (PAI-LM)  |               | 7765c381    |
| flash_attn         |               | 2.4.2       |
## SFT/DPO Training Pipeline 🔄

### 1. Model Weight Conversion (HF → Megatron) 🔄

```bash
bash scripts/train/megatron/convert/qwen2_5/convert_hf_to_mcore.sh
```
-  config in `convert_hf_to_mcore.sh`
```bash
MODEL_SIZE=${MODEL_SIZE:-"3B"}
TP=${TP:-"2"} # Tensor Parallelism
PP=${PP:-"2"} # Pipeline Parallelism
PRECISION=${PRECISION:-"bf16"}
USE_TE=${USE_TE:-"true"} # Default
MG2HF=${MG2HF:-"false"} # Default
HF_CKPT_PATH=${HF_CKPT_PATH:-"Qwen/Qwen2.5-3B-Instruct"} # HF model path
TARGET_CKPT_PATH="./mg_models/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP}" # Output Megatron model path
```

### 2. SFT/DPO Data Preprocessing 📊

```bash
bash scripts/train/megatron/preprocess/sft_conv.sh(dpo_conv.sh)
```
-  config in `sft_conv.sh/dpo_conv.sh`
```bash
INPUT_JSON=${INPUT_JSON:-"./data/dummy_sft.json"} # Input JSON path
DATA_TYPE=${DATA_TYPE:-"conversations"} # Key for conversation data
TOKENIZER=${TOKENIZER:-"Qwen2Tokenizer"} # Tokenizer class
SEQ_LEN=${SEQ_LEN:-4096} # Sequence length
OUTPUT_DATA_PREFIX=${OUTPUT_DATA_PREFIX:-"./data/dummy_sft_mg"} # Output tokenized data path
HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"} # HF model path
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-293} # Extra vocab size (1.5B: 293, 3B: 293, 7B: 421, 14B: 421, 32B: 421, 72B: 421)
TEMPLATE=${TEMPLATE:-"chatml-idsys"} # Chat Template 
```

required sft data format:
```json
[
    {
        "conversations": [
            {
                "from": "human",
                "value": "xxx"
            },
            {
                "from": "gpt",
                "value": "xxx"
            }
        ]
    }
]
```
required dpo data format:
```json
[
    {
      "chosen":[
        {
          "value":"xxx",
          "from":"human"
        },
        {
          "value":"xxx",
          "from":"gpt"
        }
      ],
      "rejected":[
        {
          "value":"xxx",
          "from":"human"
        },
        {
          "value":"xxx",
          "from":"gpt"
        }
      ],
    },
]
```

### 3. Training Configuration 🎯

```bash
bash scripts/train/megatron/train/qwen2_5/sft_conv.sh(dpo_conv.sh)
```

- Key configurations in `sft_conv.sh/dpo_conv.sh`:

```bash
# Paths
DATASET_PATH=${DATASET_PATH:-"./data/dummy_sft_mg_conversations_maxlen_4096"} # Path prefixed of tokenized data
VALID_DATASET_PATH=${VALID_DATASET_PATH:-"./data/dummy_sft_mg_conversations_maxlen_4096"} # Same as DATASET_PATH for validation
PRETRAIN_CHECKPOINT_PATH=${PRETRAIN_CHECKPOINT_PATH:-"./mg_models/Qwen2.5-3B-hf-to-mcore-te-tp2-pp2"} # Converted Megatron model path
OUTPUT_BASEPATH=${OUTPUT_BASEPATH:-"./checkpoints/sft/"} # Output path for training checkpoints

# Training Hyperparameters
MODEL_SIZE=${MODEL_SIZE:-"3B"}
BATCH_SIZE=${BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
LR=${LR:-5e-6}
MIN_LR=${MIN_LR:-0.0}
SEQ_LEN=${SEQ_LEN:-4096}
PAD_LEN=${PAD_LEN:-4096}
EPOCHS=${EPOCHS:-10}

# Parallelism Configuration (Must match with convert_hf_to_mcore.sh)
TP=${TP:-2}
PP=${PP:-2}
```

### 4. Model Weight Conversion (Megatron → HF) 🔄

```bash
bash scripts/train/megatron/convert/qwen2_5/convert_mcore_to_hf.sh(convert_mcore_to_hf_dpo.sh)
```
-  config in `convert_mcore_to_hf.sh/convert_mcore_to_hf_dpo.sh`
```bash
MODEL_SIZE=${MODEL_SIZE:-"3B"}
TP=${TP:-"2"} # Tensor Parallelism
PP=${PP:-"2"} # Pipeline Parallelism
MG_MODEL_PATH=${MG_MODEL_PATH:-"./checkpoint/sft/checkpoint"} # Checkpoint Path
HF_CKPT_PATH=${HF_CKPT_PATH:-"Qwen/Qwen2.5-3B-Instruct"} # HF model path
PRECISION=${PRECISION:-"fp32"} # Default
USE_TE=${USE_TE:-"true"} # Default
MG2HF=${MG2HF:-"true"} # Default
TARGET_CKPT_PATH="./hf_models_from_mg/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP}" # Output HF model path
```

## Troubleshooting 🔧

### Apex Compilation Issues
1. Check if required GCC/G++ are installed in conda
2. Verify if installed PyTorch matches environment CUDA version

## Note ⚠️
- Always ensure CUDA and PyTorch versions are compatible
- Keep consistent sequence lengths across preprocessing and training
- Adjust model size configurations according to your hardware capabilities

## Supported Models 📚
- Qwen2.5 series (1.5B, 3B, 7B, 14B, 32B, 72B)