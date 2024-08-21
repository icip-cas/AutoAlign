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

* [2024.8.23] We released the first version of AutoAlign, which supports XXX and XXXX. More XXX are comming soon🔥🔥🔥. 

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

*Install for Develop*

```
pip install -e .[dev]
pre-commit install
```

### 📂 Data Preparation

Covert your data to the following format and XXXX.

We also prepared two examples to facilitate your usage. [data/dummy_sft.json](data/dummy_sft.json) is for supervised finetuning and [data/dummy_dpo.json](data/dummy_dpo.json) is for RL process.
<!-- Currently, we use the format in ```data/dummy_sft.json``` for supervised finetuning and the format in ```data/dummy_dpo.json``` for RL process. -->

### 💻 Model Preparation

### 📚 Basic Alignment Toolkit

* SFT

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
autoalign-cli dpo
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
autoalign-cli serve
```

### 🛠 Automated Alignment Toolkit

The introduction and scripts for each automated alignment algorithm are stored in the [experiments](experiments) folder. （这个文件夹建议换个名字，和automated alignment相关的）

Currently, we implemented the following algorithms for automated alignment

* RLCD

一句话介绍：RLCD is XXXXX

这里给一个简单的running example，然后please refer to [experiments/rlcd_sys](experiments/rlcd_sys) for detailed XXXX.

在每个算法的文件下下面配置完整的使用readme，建议统一一下格式。需要包括（算法的介绍、数据准备，怎么使用，怎么配置详细参数等等）

### ✏️ Model Evaluation

quick start不建议写的这么复杂，可以单独在evaluation的文件夹下介绍具体的，这里只需要简单的running example，怎么配置，在哪里看结果即可

#### Objective evaluation

## Documents

Documents of this toolkit is stored at ```./docs/```.

## Evaluation
### Objective evaluation

Objective evaluation involves assessing datasets with standard answers, where processed responses can be directly compared to these standard answers according to established rules and model performances are mesured with quantitative metrics. We utilize the OpenCompass platform to conduct these evaluations.

Usage:
``` bash
autoalign-cli eval --config eval.yaml
```
In `eval.yaml`, the `model_path` is the absolute path to the evaluated model or the relative path from the root directory of this repository.

After running the above command, `autoalign-cli` will call the interface in OpenCompass to conduct an objective dataset evaluation. We format the timestamp and append it to the model_name as a directory name(`{model_id} = {model_name + timestamp}`), storing the evaluation results in the `outputs/{model_id}` directory. The raw result will be stored at `outputs/{model_id}/opencompass_log/{opencompass_timestamp}`, in which `{opencompass_timestamp}` is the default name of opencompass output directory of an evaluation. We will summarize and display each evaluation in `outputs/{model_id}/ordered_res.csv` and `outputs/{model_id}/ordered_res.txt`(formed output, easy to read).

Before starting opencompass, we will check whether new file paths exist, including the config file: `configs/{model_id}.py`, result files: `outputs/{model_id}/ordered_res.csv` and  `outputs/{model_id}/ordered_res.txt`, opencompass logs: `outputs/{model_id}/opencompass_log/`. If one of them exists, you need to choose to continue evaluating or to exit. Continuing may cause overwriting.

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

## 💳 License

This project is licensed under the [MIT License](LICENSE).
