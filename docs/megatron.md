# Megatron Training Pipeline 🚀

This repository is based on [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch.git) for secondary development, thanks to the original author's work. This project is used to support Alignment such as SFT and DPO etc. Currently, the operation flow of SFT and DPO is as follows.

## Environment Setup 🛠️

### Prerequisites
- Miniconda/Anaconda
- Git
- NVIDIA GPU with CUDA support

### Environment Configuration 🔧
```bash
bash scripts/train/megatron/env_install.sh
```
> You can also refer to the docker images(`dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07`)  provided in the Pai-Megatron-Patch repository.
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