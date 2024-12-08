# 🚀 Ascend Environment Setup Guide

## 🛠️ Step 1: Avoid Explicit flash-attention Calls

**Comment out the following lines in the `auto-alignment/src/autoalign/train` directory:**

```
train/sft.py: line 175
train/rm.py: line 189
train/dpo.py: lines 72 and 78
```

> This means we won't be using `attn_implementation`

## 📝 Step 2: Modify the pyproject.toml

**Comment out the installation of vllm and flash-attn packages:**

```toml
# "vllm==0.4.3"
# "flash-attn>=2.0"
```

## 📦 Step 3: Install Basic Packages

Run the following command:

```bash
pip install .[train]
```

## 🔧 Step 4: Install Ascend-specific Packages

Execute the following:

```bash
pip install decorator torch_npu
```

---

🎉 Congratulations! You've now set up your Ascend environment. Happy coding! 🖥️