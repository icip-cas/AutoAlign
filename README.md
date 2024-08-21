![logo](/assets/auto_logo.png)

<p align="center">
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-features">📪Features</a> •
    <a href="#-reference-results">📈Results</a> •
    <a href="#-issues">🐛Issues</a> •
    <a href="#-citation">📜Citation</a>
</p>

## 📣 About

Auto-Alignment is a package focusing on scalable and automated alignment methods. We aim to provide the academic community with a series of classic alignment baselines and ready-to-use automated alignment algorithms. This toolkit is designed to facilitate research in the field of LLM alignment.

The core functionalities of the toolkit include:

- Implementation of common model alignment algorithms (e.g., SFT, RS, RM, DPO, etc.)
- Implementation of various automatic model alignment algorithms (e.g., CAI, SPIN, RLCD, etc.)
- Efficient model sampling
- Automated model evaluation
- Post-training intervertion methods (e.g., Represenatation Engineering, Model Averaging, etc.)

*这里最好有张图，整体介绍一下这个仓库有哪些功能/组织方式*

## 🚀 News

**[2024.8.23]** We released the first version (v0.0.1) of AutoAlign, which supports CAI, PCON and a varient of RLCD. More algorithms are comming soon🔥🔥🔥.

## 🔥 Quick Start

### 🔨 Environment Setup

*Default*

```
pip install .[train]
```

*Evaluation (Optional)*

```
pip install .[eval]
bash ./scripts/post_install.sh
```

### 📂 Data Preparation

Covert your data to the following format and XXXX.

We also prepared two examples to facilitate your usage. [data/dummy_sft.json](data/dummy_sft.json) is for supervised finetuning and [data/dummy_dpo.json](data/dummy_dpo.json) is for RL process.
<!-- Currently, we use the format in ```data/dummy_sft.json``` for supervised finetuning and the format in ```data/dummy_dpo.json``` for RL process. -->

### 💻 Model Preparation

### 📚 Basic Alignment Toolkit

### SFT

这里给每个部分写一个简短的running example，包含最基本的数据和模型信息
然后说明怎么获取详细的可配置参数
DPO和inference也类似

``` bash
autoalign-cli sft
```

### Reward Modeling

```bash
autoalign-cli rm
```

### DPO

```bash
autoalign-cli dpo --model_name_or_path "Qwen2/Qwen2-7B-Instruct"  \
            --data_path "data/dummy_dpo.json" \
            --bf16 True \
            --output_dir "models/qwen2-7b-dpo" \
            --conv_template_name chatml \
            --deepspeed "configs/zero3.json"
```

### Inference

```bash
autoalign-cli infer --backend "vllm" \
            --model-name "Qwen2-0.5B-Instruct" \
            --model-path "Qwen/Qwen2-0.5B-Instruct" \
            --test-file "data/dummy_sft.json" \
            --template "chatml" \
            --source "qwen2_0_5b_instruct_dummy"
```

### Serve

```bash
autoalign-cli serve --checkpoint-path "Qwen2/Qwen2-7B-Instruct" \
                    --mode "browser" \
                    --template "chatml"
```

### 🛠 Automated Alignment Toolkit

The introduction and scripts for each automated alignment algorithm are stored in the [algorithms](./algorithms) folder.

Currently, we implemented the following algorithms for automated alignment

* RLCD

一句话介绍：RLCD is XXXXX

这里给一个简单的running example，然后please refer to [experiments/rlcd_sys](experiments/rlcd_sys) for detailed XXXX.

在每个算法的文件下下面配置完整的使用readme，建议统一一下格式。需要包括（算法的介绍、数据准备，怎么使用，怎么配置详细参数等等）

### ✏️ Model Evaluation

quick start不建议写的这么复杂，可以单独在evaluation的文件夹下介绍具体的，这里只需要简单的running example，怎么配置，在哪里看结果即可

``` bash
autoalign-cli eval --config eval.yaml
```

You can configure inference parameters in the file `eval.yaml`. For objective evaluation, the results will be displayed in `outputs/{model_id}/ordered_res.txt` at the root directory of the repository. For more information, please read `docs/eval.md`.

### Model Merging

```bash
autoalign-cli merge --model_paths "psmathur/orca_mini_v3_13b" "WizardLM/WizardLM-13B-V1.2" "garage-bAInd/Platypus2-13B" \
                    --merged_model_path "merged_model" \
                    --merging_method "average"
```

The models in `model_paths` should share the same structure.

## Documents

Documents of this toolkit is stored at ```./docs/```.

## Algorithms

Currently, the algorithms are stored in ```./experiments/```.

In the near future, we plan to integrate the representative methods into the code base.

## Evaluation
### Objective evaluation


## 📪 Features

这里可以给一个表列出现在Support的模型、数据、benchmark、算法等等。

## 📈 Reference Results

## 📅 TODO

## 📜 Citation

```bibtex
@software{AutoALign,
  author = {},
  title = {},
  url = {},
  version = {1.0.0},
  year = {2024}
}
```

## 🤝 Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

*Install for Develop*

```
pip install -e .[dev]
pre-commit install
```

## 💳 License

This project is licensed under the [MIT License](LICENSE).
