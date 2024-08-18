## Parameter Contrast

One simple weak supervision signal is that treat the response from bigger model as positive, and treat the response from smaller one as negative.

These perference pairs can be collected by running the following scripts:

```
export WEAK_MODEL_NAME=qwen2-0_5b
export WEAK_MODEL_PATH="./pretrained_models/Qwen2-0.5B"
export STRONG_MODEL_NAME=qwen2-7b
export STRONG_MODEL_PATH="./pretrained_models/Qwen2-7B"
export TEMPLATE_NAME="chatml-keep-system"
export PROMPTS_FILE="../../data/train/ultrafeedback_ata.json"
bash inference_for_pcon.sh
export OUTPUT_DIR="./outputs"
bash prepare_for_dpo.sh
```

Then start training!

```
bash train_dpo.sh
```
