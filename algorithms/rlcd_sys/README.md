# RLCD (System message)

This method use addditional system message to further constrain and improve the model behavior.

It can be viewed as a more general varient of RLCD since it use principles instead of assistant(harmful) or assistant(harmless) to form contrastive pairs.

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

Each system message pair is sampled from 4 groups of system messages in `src/autoalign/prompts/rlcd.py` mainly constraining harmlessness and helpfulness.

## Data Preparation

These perference pairs can be collected by running the following scripts:

```bash
# prepare for inference
export PROMPTS_FILE="../../data/train/ultrafeedback_ata.json"
export OUTPUT_DIR="./outputs/"
export OUTPUT_CHOSEN_FILE_NAME="ultrafeedback_ata_chosen.json"
export OUTPUT_REJECTED_FILE_NAME="ultrafeedback_ata_rejected.json"
source prepare_for_rlcd.sh
```

## Inference

Here we use a weak instruction-following model (i.e., Qwen2-7B) for context distillation:

```bash
# inference with the postive and negative prompts
export TEMPLATE_NAME="chatml"
export MODEL_NAME="qwen2-7b"
export SAVED_MODEL_PATH="./pretrained_models/Qwen2-7B"
source inference_for_rlcd.sh
bash prepare_for_dpo.sh
```
We notice a large number of responses may be identical. A model optimized for system messages might perform better. To avoid model collapse, we filter out contrastive pairs with the same response.

## Learning

After we get the contrastive pairs, we can start training!

```bash
bash train_dpo.sh
```

## Reference Performance

| Model | Dataset / Algorithm |	MT-Bench | MATH | GSM-8K | HumanEval | MBPP | HumanEval-CN | MBPP-CN | MMLU	| GPQA | CMMLU |C-Eval
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Qwen-2-7b | Base | 5.03 | 41.3 |	79.76 | 61.59 | 51 | 60.37 | 48.4 |	62.4 |	31.31 |	67.72 |	42.66
| Qwen-2-7b | Instruct | 8.15	| 25.38 | 81.35	|51.22 | 48.6 | 61.59 | 24.2 | 64.1 | 31.82 | 62.24	| 46.04
| Qwen-2-7b | Ultrachat | 7.34 | 37.98 | 77.41 |	20.73 |	34.6 | 11.59 | 32.8 | 61.35 | 31.31 | 72.23 | 63.18
| Qwen-2-7b | rlcd_sys | 7.29	|	20.76 | 52.31 |	35.98 | 36 | 29.88 | 35.4 | 52.89 | 21.21 | 68.98 | 71.35

In our reproduction, we can obtain a model with MT-Bench performance approximately equal to that of a model trained on Ultrachat.

## Citation

```
@inproceedings{
    yang2024rlcd,
    title={{RLCD}: Reinforcement Learning from Contrastive Distillation for {LM} Alignment},
    author={Kevin Yang and Dan Klein and Asli Celikyilmaz and Nanyun Peng and Yuandong Tian},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=v3XXtxWKi6}
}
```
