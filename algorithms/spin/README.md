# SPIN

## Inference

Here we use a weak instruction-following model (i.e., Qwen2-7B) for SPIN experiments:

```
export PROMPTS_FILE="../../data/ultrachat.json"
export MODEL_NAME="qwen2-7b"
export SAVED_MODEL_PATH="../../pretrained_models/Qwen2-7B"
export TEMPLATE_NAME="chatml"
export OUTPUT_DIR="./outputs"
export REJECTED_SOURCE_TAG="Qwen2-7B"
source inference_for_spin.sh
```

### Train

```

```
