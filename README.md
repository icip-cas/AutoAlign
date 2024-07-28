# Auto-Alignment

Auto-Alignment 是一个基于自动对齐技术的训练、部署和评测的大模型对齐工具包，通过提供基础和自动化的对齐算法，帮助用户使用基础模型快速对齐高质量模型

工具包的核心功能包括：
- 常见模型对齐的基础算法实现
- 多种模型自动对齐的基础算法实现
- 高效多样的模型采样
- 自动化模型评测

# Install

Default

```
pip install .[train]
```

Evaluation (Optional)

```
pip install .[eval]
```

Install for Develop

```
pip install -e .[dev]
```


## Usage

``` bash
autoalign-cli sft
autoalign-cli dpo
autoalign-cli infer
autoalign-cli eval --backend "vllm"
```

## Fine-tuning
### Data

We use sharegpt format data for supervised fine-tuning. The format are as follows:
```json
[
    {
        "id": "0",
        "conversations": [
            {
                "from": "system",
                "value": "You are a helpful artificial assistant who gives friendly responses."
            },
            {
                "from": "human",
                "value": "Tell me about Beethoven."
            },
            {
                "from": "gpt",
                "value": "Beethoven is a great composer."
            }
        ]
    },
    {
        ...
    }
]
```

## TODO

```bash
export MODEL_PATH=meta-llama/Meta-Llama-3-8B
export DATA_PATH=data/dummy_sft.json
export OUTPUT_DIR=models/llama3-sft

bash scripts/train_sft.sh
```

## Direct Preference Optimization
### Data

We use data format similar to SFT for direct preference optimization. The format are as follows:
```json
{
    "prompt": "Tell me about Beethoven." ,
    "chosen":[
        {
            "value":"Tell me about Beethoven.",
            "from": "human"
        },
        {
            "value":"Beethoven is a great composer.",
            "from": "gpt"
        }
    ],
    "rejected":[
        {
            "value":"Tell me about Beethoven.",
            "from": "human"
        },
        {
            "value":"Sorry, there is no information about Beethoven.",
            "from": "gpt"
        }
    ]
}
```


### DPO Llama-3-8B with Local GPUs

```bash
export MODEL_PATH=meta-llama/Meta-Llama-3-8B
export DATA_PATH=data/dummy_dpo.json
export OUTPUT_DIR=models/llama3-sft

bash scripts/train_dpo.sh
```


## Test

### Test Conversation Template

You can use the following script to test the newly added conversation template:

```bash
python tests/test_conversation.py test_get_tokenized_conversation \
    --template_name vicuna_v1.1 \
    --tokenizer_name_or_path meta-llama/Llama-3-8B \
    --model_max_length 4096 \
    --data_path data/dummy_sft.json
```
## Evaluation
### Objective evaluation
Objective evaluation involves assessing datasets with standard answers, where processed responses can be directly compared to these standard answers according to established rules and model performances are mesured with quantitative metrics. This project utilizes the OpenCompass platform to conduct these evaluations.

Usage:
``` bash
autoalign-cli eval --config eval.yaml
```
In `eval.yaml`, the `model_path` is the absolute path to the evaluated model or the relative path from the root directory of this repository.

After running the above command, `autoalign-cli` will call the interface in OpenCompass to conduct an objective dataset evaluation. We format the timestamp and append it to the model_name as a directory name(`{model_id} = {model_name + timestamp}`), storing the evaluation results in the `outputs/{model_id}` directory. The raw result will be stored at `outputs/{model_id}/opencompass_log/{opencompass_timestamp}`, in which `{opencompass_timestamp}` is the default name of opencompass output directory of an evaluation. We will summarize and display each evaluation in `outputs/{model_id}/ordered_res.csv` and `outputs/{model_id}/ordered_res.txt`(formed output, easy to read).

Before starting opencompass, we will check whether new file paths exist, including the config file: `configs/{model_id}.py`, result files: `outputs/{model_id}/ordered_res.csv` and  `outputs/{model_id}/ordered_res.txt`, opencompass logs: `outputs/{model_id}/opencompass_log/`. If one of them exists, you need to choose to continue evaluating or to exit. Continuing may cause overwriting.
If


## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
