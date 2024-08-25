## Parameter Contrast

One simple weak supervision signal is that treat the response from larger model as positive, and treat the response from smaller one as negative.

These perference pairs can be collected by running the following scripts:

```bash
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

```bash
bash train_dpo.sh
```

## Reference Performance

| Model | Dataset / Algorithm |	MT-Bench | MATH | GSM-8K | HumanEval | MBPP | HumanEval-CN | MBPP-CN | MMLU	| GPQA | CMMLU |C-Eval
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Qwen-2-7b | Base | 5.03 | 41.3 |	79.76 | 61.59 | 51 | 60.37 | 48.4 |	62.4 |	31.31 |	67.72 |	42.66
| Qwen-2-7b | Instruct | 8.15 | 25.38 | 81.35 | 51.22 | 48.6 | 61.59 | 24.2 | 64.1 | 31.82 | 62.24	| 46.04
| Qwen-2-7b | Ultrachat | 7.34 | 37.98 | 77.41 | 20.73 | 34.6 | 11.59 | 32.8 | 61.35 | 31.31 | 72.23 | 63.18
| Qwen-2-7b | pcon | 6.6 | 35.37 | 47.43 | 42.54 | 79.83 | 41.46 | 50.4	| 57.32 | 46.8 | 63.31 | 28.28 | 71.29 | 48.87

## Citation

```

```
