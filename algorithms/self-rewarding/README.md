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
export OUTPUT_DIR=saved_models/llama3-8b-ift-temp
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
export OUTPUT_DIR=saved_models/llama3-8b-ift-temp
export MODEL_PATH=/mnt/userdata/hf_models/NousResearch/Meta-Llama-3-8B
bash scripts/ift.sh
```
## Inference iter1

## DPO iter1
export DATA_PATH=/share/xudong2022/auto-alignment/algorithms/self-rewarding/data/eft_iter2/preference_data.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=/share/xudong2022/llama_3_8b_oasst_seed_data_eft_dpo_iter2
export MODEL_PATH=/share/xudong2022/llama_3_8b_oasst_seed_data_eft_dpo_iter1
## Inference iter2

## DPO iter2

export DATA_PATH=exp/llama3-8b/eft/iter1/preference_data.json
export CONV_TEMPLATE=llama-3-instruct
export OUTPUT_DIR=saved_models/llama3-8b-eft-dpoiter1
export MODEL_PATH=saved_models/llama3-8b-eft

## Evaluation

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

## Evaluation



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
