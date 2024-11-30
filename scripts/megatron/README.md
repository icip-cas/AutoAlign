# Megatron Training Pipeline 🚀

This repository contains scripts and instructions for training LLM  using Megatron-LM.

## Environment Setup 🛠️

### Prerequisites
- Miniconda/Anaconda
- Git
- NVIDIA GPU with CUDA support

### Environment Configuration 🔧
```bash
bash install.sh
```

## Training Pipeline 🔄

All scripts should be executed from the `megatron/` directory.

### 1. Model Weight Conversion (HF → Megatron) 🔄

```bash
export CUDA_VISIBLE_DEVICES=0
cd ../../src/autoalign/train_megatron/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
    1.5B \
    ${HF_MODEL_PATH} \
    ${OUTPUT_MG_MODEL_PATH} \
    2 \
    2 \
    bf16 \
    true \
    false
```

Parameters explained:
- Model size selection (1.5B)
- HF model path
- Output Megatron model path
- TP (Tensor Parallelism)
- PP (Pipeline Parallelism)
- Precision (bf16)
- USE_TE (true)
- MG2HF (false)

### 2. Data Preprocessing 📊

```bash
export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd ../../src/autoalign/train_megatron/toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
    ${INPUT_JSON_PATH} \
    conversations \
    Qwen2Tokenizer \
    8192 \
    ${OUTPUT_TOKENIZED_PATH} \
    ${HF_MODEL_PATH}
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

Key configurations in `train/qwen2_5/sft_conv.sh`:

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
export CUDA_VISIBLE_DEVICES=0
cd toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
    1.5B \
    ${MG_CHECKPOINT_PATH} \
    ${OUTPUT_HF_PATH} \
    2 \
    2 \
    fp32 \
    true \
    true \
    ${ORIGINAL_HF_PATH}
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