## RLCD (System message)

This method use addditional system message to further constrain and improve the model behavior.

It can be viewed as a more general varient of RLCD.

Specifically, we prompt the model to generate pair of responses via a pair of positive and negative system messages:

```
<|im_start|>system
{positive_system_message}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
```

```
<|im_start|>system
{negative_system_message}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
```

These perference pairs can be collected by running the following scripts:

```bash
# prepare for inference
export PROMPTS_FILE="../../data/train/ultrafeedback_ata.json"
export OUTPUT_DIR="./outputs/"
export OUTPUT_CHOSEN_FILE_NAME="ultrafeedback_ata_chosen.json"
export OUTPUT_REJECTED_FILE_NAME="ultrafeedback_ata_rejected.json"
source prepare_for_rlcd.sh
```

```bash
# inference with the postive and negative prompts
export TEMPLATE_NAME="chatml"
export MODEL_NAME="qwen2-7b"
export SAVED_MODEL_PATH="./pretrained_models/Qwen2-7B"
source inference_for_rlcd.sh
bash prepare_for_dpo.sh
```

Then start training!

```bash
bash train_dpo.sh
```

## Citation

```

```
