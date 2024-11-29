# Megatron Training Pipeline 🚀

This repository contains scripts and instructions for training LLM  using Megatron-LM.

## Environment Setup 🛠️

### Prerequisites
- Miniconda/Anaconda
- Git
- NVIDIA GPU with CUDA support

### Environment Configuration 🔧
```bash
# Python packages: Use pip (e.g., torch)
# System packages: Use conda (e.g., CUDA)

# Activate conda
source ~/miniconda3/bin/activate

# Create environment
conda create -n mg-re python==3.10

# Activate environment
conda activate mg-re

# Install CUDA
# Available versions: https://anaconda.org/nvidia/cuda
# Search available CUDA versions:
# conda search -c nvidia/label/cuda-12.1.1 cuda 

conda install -c nvidia/label/cuda-12.1.1 --override-channels cuda=12.1.1
# Alternative: conda install nvidia/label/cuda-12.1.1::cuda=12.1.1

# Install cuDNN (required for transformer_engine)
conda install cudnn

# Install PyTorch
# Available versions: https://pytorch.org/get-started/previous-versions/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install build tools
python3 -m pip install -U -i https://pypi.tuna.tsinghua.edu.cn/simple pip packaging ninja psutil

# Install Apex
git clone https://github.com/NVIDIA/apex apex_cu121
cd apex_cu121
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ..

# Install Flash Attention
git clone -b v2.4.2 https://github.com/Dao-AILab/flash-attention flash-attention_cu121_242
cd flash-attention_cu121
python setup.py install
cd ..

# Install Transformer Engine
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git transformer_engine_cu121
cd transformer_engine_cu121
export NVTE_FRAMEWORK=pytorch
pip install .

# Install other dependencies
pip install six transformers datasets pybind11

# Clone training repository
git clone https://github.com/jerryli1981/PAI-Megatron-LM-240718.git Megatron-LM-240718
cd Megatron-LM-240718
git checkout 7765c381d5af76f97834934a67b1e43afece02ad
cp Megatron-LM-240718 src/autoalign/train_megatron
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