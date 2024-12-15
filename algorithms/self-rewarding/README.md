# Self-rewarding

This method use language model itself to provide its own rewards during training via LLM-as-a-Judge prompting.

Specifically, the model to be optimized do the following steps iteratively:
1. Sample some instruction from an instruct model
2. Generate candidate responses to the instructions above
3. Assign rewards to its own generations via LLM-as-a-Judge prompting
4. Generate pairwise preference data according to the rewards
5. Using DPO to train on the preferences

This training both improves the instruction following capability of the model, as well as its reward-modeling ability across the iterations, which means the model is better able to assign rewards in future iterations for improving instruction following – a kind of virtuous circle.

While this improvement likely saturates in realistic scenarios, it still allows for the possibility of continual improvement beyond the human preferences that are typically used to build reward models and instruction following models today.

## IFT Seed Data Preparation

IFT seed data can be download and sampled by running the following scripts:

```bash
# Assume you are at the folder of self-rewarding
bash scripts/ift_data_prepare.sh
```

## IFT

IFT script is in the folder of `scripts/ift.sh`. You need to set the following parameters: DATA_PATH, CONV_TEMPLATE, OUTPUT_DIR, MODEL_PATH

```bash
# Assume you are at the folder of self-rewarding
export DATA_PATH=data/seed.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=saved_models/llama3-8b-ift
export MODEL_PATH=/mnt/userdata/hf_models/NousResearch/Meta-Llama-3-8B
bash scripts/ift.sh
```

## EFT Data Preparation

We need to use the model just finetuned at the last step (SFT baseline). In this step, we use the SFT baseline to generate output evaluations for each input, and accept the input-output pairs if the ranking of their scores agrees with the human rankings in the dataset.

```bash
# Assume you are at the folder of self-rewarding
bash scripts/eft_data_prepare.sh
```

## EFT

```bash
# Assume you are at the folder of self-rewarding
export DATA_PATH=data/seed.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=saved_models/llama3-8b-eft
export MODEL_PATH=/mnt/userdata/hf_models/NousResearch/Meta-Llama-3-8B
bash scripts/eft.sh
```
## Inference iter1

## DPO iter1

## Inference iter2

## DPO iter2


## Evaluation



## Reference Performance

| Model | Dataset / Algorithm |	MT-Bench | IFEval-Pr.(S) | IFEval-Ins.(S) | IFEval-Pr.(L) | IFEval-Ins.(L) | ARC-e | ARC-c | Hellaswag | SIQA | PIQA | GSM8K | MMLU | OpenBookQA | NQ
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| LLama-3-8b | Base(M0) | 1.86 | 23.48 | 35.61 | 26.43 | 39.45 | 69.84 | 45.42 | 74.68 | 46.67 | 80.96 | 55.95 | 66.62 | 50.6 | 16.09
| LLama-3-8b | IFT(SFT-baseline) | 5.46 | 36.6 | 47.12 | 41.59 | 52.76 | 74.43 | 47.46 | 76.99 | 49.23 | 82.81 | 57.24 | 66.36 | 52.6 | 29.58
| LLama-3-8b | EFT(M1) | 5.48 | 35.86 | 48.2 | 40.85	| 53.12 | 70.9 | 47.8 | 75.4 | 47.75 | 81.94 | 57.77 | 66.27 | 52 | 29.94
| LLama-3-8b | Self-Rewarding-iter1(M2) | 5.54	| 36.41 | 48.44 |	41.77 | 53.72 | 70.9 | 47.8 | 75.41 | 47.8 | 81.99 | 57.62 | 66.22 | 52.2 | 29.86
| LLama-3-8b | Self-Rewarding-iter1(M3) | 5.58	| 36.78 | 48.68 | 41.96 | 53.84 | 71.08 | 48.14	| 75.41 | 47.75 |	81.83	| 57.62 | 66.27 | 52.2 | 29.81

In our reproduction, we can obtain a model with MT-Bench and performance continuincreasing

## Citation

```
@misc{yuan2024selfrewardinglanguagemodels,
      title={Self-Rewarding Language Models},
      author={Weizhe Yuan and Richard Yuanzhe Pang and Kyunghyun Cho and Xian Li and Sainbayar Sukhbaatar and Jing Xu and Jason Weston},
      year={2024},
      eprint={2401.10020},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2401.10020},
}
```
