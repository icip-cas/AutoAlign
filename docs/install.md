# Installation Guide
### Prerequisites
- Python 3.10
- Conda (Anaconda or Miniconda)
- CUDA-compatible GPU (for flash-attn support)

### Basic Installation

Create a new conda environment and install the base package:

```bash
conda create -n ata python=3.10
conda activate ata
pip install .
pip install .[flash-attn]
```

### Optional Components

#### 📊 Evaluation Environment

For running evaluations and benchmarks:

```bash
conda create -n ata_eval --clone ata
conda activate ata_eval
pip install .[eval]
bash ./scripts/post_install.sh
```

#### 🚀 Megatron Support

For distributed training with Megatron-LM:

```bash
conda activate ata
bash scripts/train/megatron/env_install.sh
```

> **Note:** Some version conflicts may occur during Megatron installation. These can typically be safely ignored if the installation completes successfully.

### Verification

After installation, verify your setup:

```bash
# Check if the package is installed
python -c "import autoalign; print('AutoAlign successfully installed!')"

# Or check with pip
pip show autoalign
```