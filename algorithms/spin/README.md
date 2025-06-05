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

## Data Preparation

```
export CHOSEN_FILE="../../data/ultrachat.json"
export CHOSEN_SOURCE_TAG="golden"
source prepare_for_dpo.sh
```

### Train

```bash
bash train_dpo.sh
```

## Citation

```
@misc{chen2024selfplay,
      title={Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models}, 
      author={Zixiang Chen and Yihe Deng and Huizhuo Yuan and Kaixuan Ji and Quanquan Gu},
      year={2024},
      eprint={2401.01335},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```