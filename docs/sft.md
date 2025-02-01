# Supervise Finetuning

This experiment is used for supervised fine-tuning (SFT) on UltraChat. This README provides instructions for setting up and running the experiment.

### Data Preparation

To begin, you'll need to obtain the following datasets:

1. [UltraChat Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)

### Supervised Fine-Tuning (SFT)

To perform Supervised Fine-Tuning on the UltraChat dataset, use the following script:

```
export DATA_PATH=data/ultrachat.json
export CONV_TEMPLATE=llama-3-instruct
export MODEL_PATH=pretrained_models/Meta-Llama-3-8B
export GA=16
export OUTPUT_DIR=saved_models/llama-3-8b_ultrachat
bash scripts/train_sft.sh
```

### References

```
@article{ding2023enhancing,
  title={Enhancing Chat Language Models by Scaling High-quality Instructional Conversations},
  author={Ding, Ning and Chen, Yulin and Xu, Bokai and Qin, Yujia and Zheng, Zhi and Hu, Shengding and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen},
  journal={arXiv preprint arXiv:2305.14233},
  year={2023}
}

```