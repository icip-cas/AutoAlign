# Megatron Training Pipeline 🚀

This repository is based on [alibaba/Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch.git) for secondary development, thanks to the original author's work. This project is used to support Alignment such as SFT and DPO etc. Currently, the operation flow of SFT is as follows, and the detailed introduction of DPO is coming soon.

## Environment Setup 🛠️

### Prerequisites
- Miniconda/Anaconda
- Git
- NVIDIA GPU with CUDA support

### Environment Configuration 🔧
```bash
bash megatron_autoalign/scripts_megatron/env_install.sh
```
> You can also refer to the docker images(`dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07`)  provided in the Pai-Megatron-Patch repository.
## Training Pipeline 🔄

All scripts should be executed from the `megatron_autoalign/` directory.

### 1. Model Weight Conversion (HF → Megatron) 🔄

```bash
export CUDA_VISIBLE_DEVICES=0
cd AutoAlign/megatron_autoalign/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
    1.5B \ # Model size selection (1.5B)
    ${HF_MODEL_PATH} \ # HF model path
    ${OUTPUT_MG_MODEL_PATH} \ # Output Megatron model path
    2 \ # TP (Tensor Parallelism)
    2 \ # PP (Pipeline Parallelism)
    bf16 \ # Precision (bf16)
    true \ # USE_TE (default:true)
    false # MG2HF (default:false)
```

### 2. Data Preprocessing 📊

```bash
export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd AutoAlign/megatron_autoalign/toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
    ${INPUT_JSON_PATH} \ # Input JSON path
    conversations \ # Key for conversation data
    Qwen2Tokenizer \ # Tokenizer class
    8192 \ # Sequence length
    ${OUTPUT_TOKENIZED_PATH} \ # Output tokenized data path
    ${HF_MODEL_PATH} \ # HF model path
    ${EXTRA_VOCAB_SIZE} \ # Extra vocab size
    ${TEMPLATE} # Chat Template 
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

Key configurations in `AutoAlign/megatron_autoalign/scripts_megatron/train/qwen2_5/sft_conv.sh`:

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
cd AutoAlign/megatron_autoalign/toolkits/model_checkpoints_convertor/qwen
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