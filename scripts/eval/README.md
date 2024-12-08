### 介绍
由于依赖包可能的不稳定性, 默认使用单模型评测，若需要多模型评测，请自定义opencompass config文件。

#### 简单版能力评测
- 数学: GSM8K
- 代码: MBPP
- 知识: MMLU
- 综合能力: BBH
- 通用能力: ApacaEval

#### 完全版能力评测
- 数学: MATH, GSM8K,
- 代码: HumanEval, MBPP,
- 知识: MMLU, CMMLU, GPQA
- 综合能力: BBH, C-Eval
- 通用能力: ApacaEval, Mtbench, AlignBench

### 使用方法
#### 一键启动
```shell
bash eval.sh {model_name} {model_path} {batch_size} {num_gpus}
# 若启动主观评测，请配置OPENAI_BASE_URL与OPENAI_API_KEY,否则
export OPENAI_BASE_URL="你的openai代理地址，若可以访问openai则不需要配置"
export OPENAI_API_KEY="你的openai api key"
# 单独启动主观评测
bash eval_openai.sh $model_name $model_path
# 单独启动客观评测
bash eval_opencompass.sh $model_name $model_path $eval_type $batch_size $num_gpus
```

#### 安装依赖与评测数据(一键安装)
(若安装失败，请参考"手动安装依赖与评测数据"进行安装)
```shell
bash build_eval_package.sh
```

#### 手动评测
默认使用vllm进行客观benchmark的评测，若不使用，请参考下面的代码自行配置
```shell
python opencompass/run.py --datasets gsm8k_gen math_gen bbh_gen mmlu_gen agieval_gen hellaswag_gen \
--hf-path $model_path \
--model-kwargs device_map='auto' use_flash_attention_2=True \
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \
--max-seq-len 2048 \
--max-out-len 200 \
--batch-size 8 \
--num-gpus 1
```

并下载opencompass数据集，解压到{opencompass root path}/data目录下

```shell
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
mv {dict} opencompass/data
```
