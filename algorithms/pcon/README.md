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
@inproceedings{kim-etal-2023-aligning,
    title = "Aligning Large Language Models through Synthetic Feedback",
    author = "Kim, Sungdong  and
      Bae, Sanghwan  and
      Shin, Jamin  and
      Kang, Soyoung  and
      Kwak, Donghyun  and
      Yoo, Kang  and
      Seo, Minjoon",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.844/",
    doi = "10.18653/v1/2023.emnlp-main.844",
    pages = "13677--13700",
    abstract = "Aligning large language models (LLMs) to human values has become increasingly important as it enables sophisticated steering of LLMs. However, it requires significant human demonstrations and feedback or distillation from proprietary LLMs such as ChatGPT. In this work, we propose a novel alignment learning framework with synthetic feedback not dependent on extensive human annotations and proprietary LLMs. First, we perform reward modeling (RM) with synthetic feedback by contrasting responses from vanilla LLMs with various sizes and prompts. Then, we use the RM to simulate high-quality demonstrations to train a supervised policy and further optimize the model with reinforcement learning. Our resulting model, Aligned Language Model with Synthetic Training dataset (ALMoST), outperforms recent open-sourced models, which are trained on the outputs of InstructGPT or human-annotated demonstrations, in alignment benchmarks. In human evaluation, our model is preferred to Alpaca and Dolly-v2, 55.0{\%} and 58.5{\%} of the time, respectively. Further analyses demonstrate the efficacy and importance of synthetic feedback in our framework."
}
```
