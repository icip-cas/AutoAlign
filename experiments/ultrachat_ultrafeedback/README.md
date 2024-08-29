## UltraChat UltraFeedback

This experiment combines supervised fine-tuning (SFT) on UltraChat and direct preference optimization (DPO) on UltraFeedback. This README provides instructions for setting up and running the experiment.

### Data Preparation

To begin, you'll need to obtain the following datasets:

1. [UltraChat Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
2. [UltraFeedback Binarized Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)


We provide a convenient script to obtain the UltraFeedback dataset in a specific format. To use it, run:

```
python get_ultrafeedback.py
```

### Supervised Fine-Tuning (SFT)

To perform Supervised Fine-Tuning on the UltraChat dataset, use the following script:

```
export DATA_PATH=data/ultrachat.json
export CONV_TEMPLATE=llama-3-instruct
export MODEL_PATH=pretrained_models/Meta-Llama-3-8B
export GA=16
export OUTPUT_DIR=saved_models/llama-3-8b_ultrachat`
bash scripts/train_sft.sh
```

### Direct Preference Optimization (DPO)

After completing the Supervised Fine-Tuning step, you can proceed with Direct Preference Optimization using the UltraFeedback dataset. Use the following script to run DPO:

```
export DATA_PATH=data/ultrafeedback.json
export CONV_TEMPLATE=llama-3-instruct
export MODEL_PATH=saved_models/llama-3-8b_ultrachat
export GA=16
export OUTPUT_DIR=saved_models/llama-3-8b_ultrachat-ultrafeedback
bash scripts/train_dpo.sh
```

### References

```
@article{ding2023enhancing,
  title={Enhancing Chat Language Models by Scaling High-quality Instructional Conversations},
  author={Ding, Ning and Chen, Yulin and Xu, Bokai and Qin, Yujia and Zheng, Zhi and Hu, Shengding and Liu, Zhiyuan and Sun, Maosong and Zhou, Bowen},
  journal={arXiv preprint arXiv:2305.14233},
  year={2023}
}

@misc{cui2023ultrafeedback,
      title={UltraFeedback: Boosting Language Models with High-quality Feedback}, 
      author={Ganqu Cui and Lifan Yuan and Ning Ding and Guanming Yao and Wei Zhu and Yuan Ni and Guotong Xie and Zhiyuan Liu and Maosong Sun},
      year={2023},
      eprint={2310.01377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}


```