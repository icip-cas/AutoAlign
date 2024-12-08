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
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
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


PATCH_PACKAGE_NAME="megatron_patch"
PATCH_INSTALL_PATH="$SITE_PACKAGES/$PATCH_PACKAGE_NAME"

print_section "COPYING Pai-Megatron-Patch/megatron_patch TO site-packages"


if [ -d "$PATCH_INSTALL_PATH" ]; then
    print_section "Removing existing Pai-Megatron-Patch/megatron_patch in site-packages"
    rm -rf "$PATCH_INSTALL_PATH"
fi


mkdir -p "$PATCH_INSTALL_PATH"


cp -r Pai-Megatron-Patch/megatron_patch/* "$PATCH_INSTALL_PATH/"
if [ $? -ne 0 ]; then
    print_error "Failed to copy Pai-Megatron-Patch to site-packages"
    exit 1
fi


find "$PATCH_INSTALL_PATH" -type d -exec touch {}/__init__.py \;

print_success "Pai-Megatron-Patch has been manually installed to site-packages"
rm -rf Pai-Megatron-Patch