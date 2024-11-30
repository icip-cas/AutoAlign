#!/bin/bash

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Helper functions
print_section() {
    echo -e "\n${BLUE}${BOLD}══════════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}${BOLD}$1${NC}"
    echo -e "${BLUE}${BOLD}══════════════════════════════════════════════════════════════${NC}\n"
}

print_error() {
    echo -e "${RED}${BOLD}ERROR: $1${NC}"
}

print_success() {
    echo -e "${GREEN}${BOLD}SUCCESS: $1${NC}"
}

# Get current path
CURRENT_PATH=$(dirname "$(readlink -f "$0")")

# Check conda environment
print_section "CHECKING ENVIRONMENT"
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_error "Please activate conda environment first"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" = "3.10" ] && [ "$PYTHON_VERSION" != "3.10" ]; then
    print_error "Python 3.10 or higher is required, current version is $PYTHON_VERSION"
    exit 1
fi

print_success "Environment check passed"

# Main installation process
print_section "STARTING MEGATRON ENVIRONMENT INSTALLATION"

# Install CUDA and cuDNN
print_section "1. INSTALLING CUDA AND CUDNN"
echo "Installing CUDA 12.1.1..."
conda install -y -c nvidia/label/cuda-12.1.1 --override-channels cuda=12.1.1
if [ $? -ne 0 ]; then
    print_error "CUDA installation failed"
    exit 1
fi

echo "Installing cuDNN..."
conda install -y cudnn
if [ $? -ne 0 ]; then
    print_error "cuDNN installation failed"
    exit 1
fi
print_success "CUDA and cuDNN installation completed"

# Install PyTorch
print_section "2. INSTALLING PYTORCH"
pip install --force-reinstall torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
print_success "PyTorch installation completed"

# Install dependencies
print_section "3. INSTALLING BUILD TOOLS AND DEPENDENCIES"
pip install -U pip packaging ninja psutil six transformers datasets pybind11
print_success "Dependencies installation completed"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Install Nvidia Apex
print_section "4. INSTALLING NVIDIA APEX"
git clone https://github.com/NVIDIA/apex apex_cu121
cd apex_cu121
git checkout ac8214ee6ba77c0037c693828e39d83654d25720
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
cd ..
print_success "Nvidia Apex installation completed"

# Install Flash Attention
print_section "5. INSTALLING FLASH ATTENTION"
git clone -b v2.4.2 https://github.com/Dao-AILab/flash-attention flash-attention_cu121_242
cd flash-attention_cu121_242
python setup.py install
cd ..
print_success "Flash Attention installation completed"

# Install Transformer Engine
print_section "6. INSTALLING TRANSFORMER ENGINE"
export NVTE_FRAMEWORK=pytorch
echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git transformer_engine_cu121
cd transformer_engine_cu121
git checkout 7a7225c403bc704264e7cf437369594aeb8b3ba3
git submodule update --init --recursive
pip install .
cd ..
print_success "Transformer Engine installation completed"

Setup Megatron-LM
print_section "7. SETTING UP MEGATRON-LM"
git clone https://github.com/jerryli1981/PAI-Megatron-LM-240718.git Megatron-LM-240718
cd Megatron-LM-240718
git checkout 7765c381d5af76f97834934a67b1e43afece02ad
cd ..

# Copy Megatron files
INTERSECTION="auto-alignment"
BASE_PATH=${CURRENT_PATH%$INTERSECTION*}$INTERSECTION

MEGATRON_DIR="$BASE_PATH/src/autoalign/train_megatron"
echo -e "${BOLD}Copying Megatron files to $MEGATRON_DIR...${NC}"
mkdir -p "$MEGATRON_DIR"
cp -r Megatron-LM-240718 "$MEGATRON_DIR/"
print_success "Megatron-LM setup completed"

# Cleanup
cd ..
rm -rf $TEMP_DIR

print_section "INSTALLATION COMPLETE"
print_success "Megatron environment installation completed successfully!"
echo -e "${BOLD}You can now proceed with using Megatron-LM${NC}"