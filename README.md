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

If 


## Contributing

If you would like to contribute to this project, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
