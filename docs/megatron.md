# Megatron Training Pipeline 🚀

This repository is based on [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch.git) for secondary development, thanks to the original author's work. This project is used to support Alignment such as SFT and DPO etc. Currently, the operation flow of SFT is as follows, and the detailed introduction of DPO is coming soon.

## Environment Setup 🛠️

### Prerequisites
- Miniconda/Anaconda
- Git
- NVIDIA GPU with CUDA support

### Environment Configuration 🔧
```bash
bash AutoAlign/scripts/train/megatron/env_install.sh
```
> You can also refer to the docker images(`dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07`)  provided in the Pai-Megatron-Patch repository.
## Training Pipeline 🔄

All scripts should be executed from the `megatron_autoalign/` directory.

### 1. Model Weight Conversion (HF → Megatron) 🔄

```bash
bash AutoAlign/scripts/train/megatron/convert/qwen2_5/convert_hf_to_mcore.sh
```
> config in convert_hf_to_mcore.sh
```bash
HF_MODELS=${HF_MODELS:-"${ROOT}/hf_models"} # HF model path
MODEL_SIZE=${MODEL_SIZE:-"7B"}
TP=${TP:-"2"}  # Tensor Parallelism
PP=${PP:-"2"}  # Pipeline Parallelism
PRECISION=${PRECISION:-"bf16"}
USE_TE=${USE_TE:-"true"}
MG2HF=${MG2HF:-"false"}
HF_CKPT_PATH=${HF_CKPT_PATH:-"${ROOT}/hf_models/Qwen2.5-${MODEL_SIZE}"}
SOURCE_CKPT_PATH=${HF_MODELS}/Qwen2.5-${MODEL_SIZE} 
TARGET_CKPT_PATH=${ROOT}/mg_models/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP} # Output Megatron model path
```

### 2. Data Preprocessing 📊

```bash
bash AutoAlign/scripts/train/megatron/preprocess/sft_conv.sh
```
> config in convert_hf_to_mcore.sh
```bash
INPUT_JSON=${INPUT_JSON:-"${ROOT}/data/dummy_sft.json"} # Input JSON path
DATA_TYPE=${DATA_TYPE:-"conversations"} # Key for conversation data
TOKENIZER=${TOKENIZER:-"Qwen2Tokenizer"} # Tokenizer class
SEQ_LEN=${SEQ_LEN:-4096} # Sequence length
OUTPUT_DATA_PREFIX=${OUTPUT_DATA_PREFIX:-"${ROOT}/data/megatron/dummy_sft"} # Output tokenized data path
HF_MODEL_PATH=${HF_MODEL_PATH:-"${ROOT}/hf_models/Qwen2.5-7B"} # HF model path
EXTRA_VOCAB_SIZE=${EXTRA_VOCAB_SIZE:-421} # Extra vocab size
TEMPLATE=${TEMPLATE:-"chatml-idsys"} # Chat Template 
```

Required data format:
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

### 3. Training Configuration 🎯

Key configurations in `AutoAlign/scripts/train/megatron/train/qwen2_5/sft_conv.sh`:

```bash
# Paths
DATASET_PATH=${TOKENIZED_DATA_PATH}
PRETRAIN_CHECKPOINT_PATH=${MG_MODEL_PATH}
OUTPUT_BASEPATH=${OUTPUT_CHECKPOINT_PATH}

# Hardware Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NNODES=1
GPUS_PER_NODE=8

# Training Parameters
MODEL_SIZE=1.5B
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
LR=1e-5
MIN_LR=1e-6
SEQ_LEN=8192
PAD_LEN=8192

# Parallel Configuration
TP=2
PP=2
SP=false
CP=1

# Dataset Configuration
dataset_option=" \
    --data-path ${DATASET_PATH} \
    --split 100,0,0 \
    --dataset mmap  \
    --epochs 1 "
```

### 4. Model Weight Conversion (Megatron → HF) 🔄

```bash
bash AutoAlign/scripts/train/megatron/convert/qwen2_5/convert_mcore_to_hf.sh
bash hf2mcore_qwen2.5_convertor.sh \
    1.5B \ # Model size selection (1.5B)
    ${MG_CHECKPOINT_PATH} \ # Megatron model path
    ${OUTPUT_HF_PATH} \ # Output HF model path
    2 \ # TP (Tensor Parallelism)
    2 \ # PP (Pipeline Parallelism)
    fp32 \ # Precision (fp32)
    true \ # USE_TE (default:true)
    true \ # MG2HF (default:false)
    ${ORIGINAL_HF_PATH} # Original HF model path
```

> config in convert_hf_to_mcore.sh
```bash
MODEL_SIZE=${MODEL_SIZE:-"7B"}
TP=${TP:-"2"}  # Tensor Parallelism
PP=${PP:-"2"}  # Pipeline Parallelism
MG_MODEL_PATH=${MG_MODEL_PATH:-"${ROOT}/mg_models/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP}"} # Megatron model path
HF_MODEL_PATH=${HF_MODEL_PATH:-"${ROOT}/hf_models/Qwen2.5-${MODEL_SIZE}-mcore-to-hf-tp${TP}-pp${PP}"} # Output HF model path
PRECISION=${PRECISION:-"fp32"}
USE_TE=${USE_TE:-"true"}
MG2HF=${MG2HF:-"true"}
HF_MODEL_SOURCE=${HF_MODEL_SOURCE:-"${ROOT}/hf_models/Qwen2.5-${MODEL_SIZE}"}  # Original HF model path
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
- Qwen2.5 series (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)