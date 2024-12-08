## DPO

This algorithm is used for direct preference optimization (DPO) on UltraFeedback. This README provides instructions for setting up and running the experiment.

### Data Preparation

To begin, you'll need to obtain the following datasets:

1. [UltraFeedback Binarized Dataset](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)


We provide a convenient script to obtain the UltraFeedback dataset in a specific format. To use it, run:

```
python get_ultrafeedback.py
```

### Direct Preference Optimization (DPO)

Equipped with an instruction-following model (eg. a model trained on UltraChat using the SFT algorithm), you can proceed with Direct Preference Optimization using the UltraFeedback dataset. To run DPO, use the following script:

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
@misc{cui2023ultrafeedback,
      title={UltraFeedback: Boosting Language Models with High-quality Feedback}, 
      author={Ganqu Cui and Lifan Yuan and Ning Ding and Guanming Yao and Wei Zhu and Yuan Ni and Guotong Xie and Zhiyuan Liu and Maosong Sun},
      year={2023},
      eprint={2310.01377},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{rafailov2024direct,
  title={Direct preference optimization: Your language model is secretly a reward model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D and Ermon, Stefano and Finn, Chelsea},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```