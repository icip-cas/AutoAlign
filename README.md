# Auto-Alignment

Auto-Alignment is a package focusing on scalable and automated alignment methods. We aim to provide the academic community with a series of classic alignment baselines and ready-to-use automated alignment algorithms. This toolkit is designed to facilitate research in the field of LLM alignment.

The core functionalities of the toolkit include:

- Implementation of common model alignment algorithms (e.g., SFT, RS, RM, DPO, etc.)
- Implementation of various automatic model alignment algorithms (e.g., CAI, SPIN, RLCD, etc.)
- Efficient model sampling
- Automated model evaluation
- Post-training intervertion methods (e.g., Represenatation Engineering, Model Averaging, etc.)

# Install

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


## Quick Start

We recommand to use huggingface mirror for Chinese users via:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### SFT

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

## Data Formatting

Currently, we use the format in ```data/dummy_sft.json``` for supervised finetuning and the format in ```data/dummy_dpo.json``` for RL process.

## Documents

Documents of this toolkit is stored at ```./docs/```.

## Algorithms

Currently, the algorithms are stored in ```./experiments/```.

In the near future, we plan to integrate the representative methods into the code base.

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


## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
