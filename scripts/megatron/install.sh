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

# Check gcc version and make
print_section "CHECKING COMPILER AND BUILD TOOLS"

# Check if gcc is installed
if ! command -v gcc >/dev/null 2>&1; then
    print_error "gcc is not installed. Installing gcc..."
    conda install -y 'gcc<13' -c conda-forge
    conda install -y 'gxx<13' -c conda-forge
    if [ $? -ne 0 ]; thenfrom megatron_pai import megatron_patch as megatron_patc
        print_error "gcc installation failed"
        exit 1
    fi
fi

# Check gcc version
GCC_VERSION=$(gcc -dumpversion | cut -f1 -d.)
if [ "$GCC_VERSION" -lt 5 ] || [ "$GCC_VERSION" -ge 13 ]; then
    print_error "gcc version $GCC_VERSION is not between 5 and 13. Installing gcc..."
    conda install -y 'gcc<13' -c conda-forge
    conda install -y 'gxx<13' -c conda-forge
    if [ $? -ne 0 ]; then
        print_error "gcc installation failed"
        exit 1
    fi
fi

# Check if make is installed
if ! command -v make >/dev/null 2>&1; then
    print_error "make is not installed. Installing make..."
    conda install -y make
    if [ $? -ne 0 ]; then
        print_error "make installation failed"
        exit 1
    fi
fi

print_success "Compiler and build tools check passed"

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
if [ $? -ne 0 ]; then
    print_error "PyTorch installation failed"
    exit 1
fi
print_success "PyTorch installation completed"

# Install dependencies
print_section "3. INSTALLING BUILD TOOLS AND DEPENDENCIES"
pip install -U pip packaging ninja psutil six transformers datasets pybind11
if [ $? -ne 0 ]; then
    print_error "Dependencies installation failed"
    exit 1
fi
print_success "Dependencies installation completed"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Install Nvidia Apex
print_section "4. INSTALLING NVIDIA APEX"
git clone https://github.com/NVIDIA/apex apex_cu121
if [ $? -ne 0 ]; then
    print_error "Failed to clone NVIDIA Apex"
    exit 1
fi
cd apex_cu121
git checkout ac8214ee6ba77c0037c693828e39d83654d25720
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
if [ $? -ne 0 ]; then
    print_error "NVIDIA Apex installation failed"
    exit 1
fi
cd ..
print_success "NVIDIA Apex installation completed"

# Install Flash Attention
print_section "5. INSTALLING FLASH ATTENTION"
git clone -b v2.4.2 https://github.com/Dao-AILab/flash-attention flash-attention_cu121_242
if [ $? -ne 0 ]; then
    print_error "Failed to clone Flash Attention"
    exit 1
fi
cd flash-attention_cu121_242
python setup.py install
if [ $? -ne 0 ]; then
    print_error "Flash Attention installation failed"
    exit 1
fi
cd ..
print_success "Flash Attention installation completed"

# Install Transformer Engine
print_section "6. INSTALLING TRANSFORMER ENGINE"
export NVTE_FRAMEWORK=pytorch
echo "NVTE_FRAMEWORK=$NVTE_FRAMEWORK"
git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git transformer_engine_cu121
if [ $? -ne 0 ]; then
    print_error "Failed to clone Transformer Engine"
    exit 1
fi
cd transformer_engine_cu121
git checkout 7a7225c403bc704264e7cf437369594aeb8b3ba3
git submodule update --init --recursive
pip install .
if [ $? -ne 0 ]; then
    print_error "Transformer Engine installation failed"
    exit 1
fi
cd ..
print_success "Transformer Engine installation completed"

# Setup Megatron-LM by cloning and copying to site-packages
git clone https://github.com/jerryli1981/PAI-Megatron-LM-240718.git Megatron-LM
if [ $? -ne 0 ]; then
    print_error "Failed to clone Megatron-LM repository"
    exit 1
fi
cd Megatron-LM
git checkout 7765c381d5af76f97834934a67b1e43afece02ad
cd ..


SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")


MEGATRON_PACKAGE_NAME="megatron"
MEGATRON_INSTALL_PATH="$SITE_PACKAGES/$MEGATRON_PACKAGE_NAME"

print_section "COPYING Megatron-LM TO site-packages"


if [ -d "$MEGATRON_INSTALL_PATH" ]; then
    print_section "Removing existing Megatron-LM in site-packages"
    rm -rf "$MEGATRON_INSTALL_PATH"
fi


mkdir -p "$MEGATRON_INSTALL_PATH"


cp -r Megatron-LM/megatron/* "$MEGATRON_INSTALL_PATH/"
if [ $? -ne 0 ]; then
    print_error "Failed to copy Megatron-LM to site-packages"
    exit 1
fi

touch "$MEGATRON_INSTALL_PATH/__init__.py"

rm -rf Megatron-LM
print_success "Megatron-LM has been manually installed to site-packages"


# Clone and install Pai-Megatron-Patch by copying to site-packages
print_section "8. CLONING AND INSTALLING PAI-MEGATRON-PATCH MANUALLY"

git clone https://github.com/alibaba/Pai-Megatron-Patch.git Pai-Megatron-Patch
if [ $? -ne 0 ]; then
    print_error "Failed to clone Pai-Megatron-Patch repository"
    exit 1
fi
cd Pai-Megatron-Patch
git checkout 9b88cc46653e2c4f7f99529f86f8737ac1da9e9a
cd ..


PATCH_PACKAGE_NAME="megatron_pai"
PATCH_INSTALL_PATH="$SITE_PACKAGES/$PATCH_PACKAGE_NAME"

print_section "COPYING Pai-Megatron-Patch TO site-packages"


if [ -d "$PATCH_INSTALL_PATH" ]; then
    print_section "Removing existing Pai-Megatron-Patch in site-packages"
    rm -rf "$PATCH_INSTALL_PATH"
fi


mkdir -p "$PATCH_INSTALL_PATH"


cp -r Pai-Megatron-Patch/* "$PATCH_INSTALL_PATH/"
if [ $? -ne 0 ]; then
    print_error "Failed to copy Pai-Megatron-Patch to site-packages"
    exit 1
fi



find "$PATCH_INSTALL_PATH" -type d -exec touch {}/__init__.py \;
print_success "Pai-Megatron-Patch has been manually installed to site-packages"

# Cleanup
cd ..
rm -rf $TEMP_DIR

print_section "INSTALLATION COMPLETE"
print_success "Megatron environment installation completed successfully!"
echo -e "${BOLD}You can now proceed with using Megatron-LM${NC}"