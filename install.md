# 安装指南 - [AutoAlign]

  

  

欢迎使用 [AutoAlign]！本文档将指导你完成项目的安装过程。本项目支持在 NVIDIA GPU 和 华为昇腾 (Ascend) NPU 两种硬件上运行。

  

  

## 目录

[安装指南 - \[AutoAlign\]](#安装指南---autoalign)

[目录](#目录)

[先决条件](#先决条件)

- [NVIDIA GPU](#nvidia-gpu)

- [Huawei Ascend NPU](#huawei-ascend-npu)

[AutoAlign安装](#autoalign安装)

- [安装指南 - \[AutoAlign\]](#安装指南---autoalign)
  - [目录](#目录)
  - [先决条件](#先决条件)
    - [NVIDIA GPU](#nvidia-gpu)
    - [Huawei Ascend NPU](#huawei-ascend-npu)
  - [AutoAlign安装](#autoalign安装)
    - [NVIDIA GPU](#nvidia-gpu-1)
      - [方法一：使用包管理器 (暂未支持)](#方法一使用包管理器-暂未支持)
      - [方法二：使用源码安装](#方法二使用源码安装)
        - [默认](#默认)
        - [评估(可选)](#评估可选)
    - [Huawei Ascend NPU](#huawei-ascend-npu-1)
      - [方法一：使用源码安装](#方法一使用源码安装)
        - [依赖一：NPU驱动](#依赖一npu驱动)
        - [依赖二：NPU相关开发包](#依赖二npu相关开发包)
- [Create a sampling params object.](#create-a-sampling-params-object)
- [Create an LLM.](#create-an-llm)
- [Generate texts from the prompts.](#generate-texts-from-the-prompts)
- [Try `export VLLM_USE_MODELSCOPE=true` and `pip install modelscope`](#try-export-vllm_use_modelscopetrue-and-pip-install-modelscope)
- [to speed up download if huggingface is not reachable.](#to-speed-up-download-if-huggingface-is-not-reachable)
        - [依赖三：AutoAlign及其依赖](#依赖三autoalign及其依赖)


## 先决条件

请根据您的硬件平台，确保满足相应的先决条件。

### NVIDIA GPU

*  **操作系统**:

Linux (例如 Ubuntu 20.04/22.04)

  

*  **硬件**:

支持 CUDA 的 NVIDIA GPU

  

*  **软件依赖**:

**NVIDIA 驱动**: 确保已安装兼容的驱动。运行 `nvidia-smi` 应能看到 GPU 信息。

```bash

nvidia-smi

```

*  **[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)**: 版本 `11.8` 或更高版本。

  

*  **[Git](https://git-scm.com/)**: 用于克隆本仓库。

  
  

### Huawei Ascend NPU

*  **操作系统**:

Linux (例如 Ubuntu, EulerOS, CentOS)

  

*  **硬件**:

华为昇腾 (Ascend) NPU

  

*  **软件依赖**:

**NPU 驱动与固件**: 确保已安装与 CANN 版本匹配的驱动和固件。运行 `npu-smi info` 应能看到 NPU 信息。

```bash

npu-smi info

```

*  **[CANN (Compute Architecture for Neural Networks)](https://www.hiascend.com/developer/download)**: 版本 `8.1.RC1`。请按照官方指引完成安装并配置环境变量。

  

*  **[Python](https.www.python.org/)**: 版本 `3.10`

  

*  **[Git](https://git-scm.com/)**: 用于克隆本仓库。

  

## AutoAlign安装

  

### NVIDIA GPU

我们推荐使用**从源码安装**的方式，因为它能确保您获得最新的代码并能灵活配置环境。

  

#### 方法一：使用包管理器 (暂未支持)

  

>  **注意**: 目前 AutoAlign 尚未发布到 PyPI。此方法将在未来版本中提供支持。

```bash

pip  install  autoalign

```

#### 方法二：使用源码安装

  

在使用源码安装AutoAlign之前，首先创建虚拟环境并且将AutoAlign 克隆到本地

```

git clone --depth 1 https://github.com/icip-cas/AutoAlign.git

conda create -n ata python=3.10

conda activate ata

```

运行以下指令以安装AutoAlign及其依赖。

##### 默认

```

cd AutoAlign

pip install .

pip install .[flash-attn]

```

##### 评估(可选)

···

conda create -n ata_eval --clone ata

conda activate ata_eval

pip install .[eval]

bash ./scripts/post_install.sh

···

### Huawei Ascend NPU

目前AutoAlign通过torch-npu库完成了对于华为昇腾910B系列的支持，与NVIDIA相比，需要额外的条件

  

1. 加速卡本身的驱动正常安装

2. CANN Toolkit和Kernels库正常安装，且CANN版本为`8.1.RC1`，请按照[安装CANN软件包](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)进行安装或升级

  

#### 方法一：使用源码安装

##### 依赖一：NPU驱动

根据昇腾卡型号安装对应的固件和驱动，可参考 [快速安装昇腾环境](https://ascend.github.io/docs/sources/ascend/quick_install.html) 指引，使用 `npu-smi info` 验证

##### 依赖二：NPU相关开发包
| Requirement | Recommend                         |
| ----------- | --------------------------------- |
| CANN        | 8.1.RC1                           |
| torch       | 2.5.1                             |
| torch-npu   | torch-npu-2.5.1.post1.dev20250528 |
| vllm        | 0.9.1                             |
| vllm-ascend | 0.9.1rc1                          |
* 请按照[安装CANN软件包](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)，[安装Pytorch](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)中的指引安装对应版本的CANN和torch。

* 随后使用下方命令安装torch-npu
```
# For torch-npu dev version or x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
pip install "torch-npu==torch-npu-2.5.1.post1.dev20250528"
``` 

* 参考[VLLM-Ascend安装文档](https://vllm-ascend.readthedocs.io/en/latest/installation.html)安装vllm，也可以使用以下命令直接安装：

```

pip install vllm==0.9.1

pip install vllm-ascend==0.9.1rc1

```

当以上依赖安装完成后，可以通过下方python脚本对torch-npu的可用情况做一下校验，预期结果为`True`：

```

import torch

import torch_npu

print(torch.npu.is_available())

```
可以使用以下代码验证vllm安装
···
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```
然后使用
```
# Try `export VLLM_USE_MODELSCOPE=true` and `pip install modelscope`
# to speed up download if huggingface is not reachable.
python example.py
···
其结果应当如下所示：
```
INFO 02-18 08:49:58 __init__.py:28] Available plugins for group vllm.platform_plugins:
INFO 02-18 08:49:58 __init__.py:30] name=ascend, value=vllm_ascend:register
INFO 02-18 08:49:58 __init__.py:32] all available plugins for group vllm.platform_plugins will be loaded.
INFO 02-18 08:49:58 __init__.py:34] set environment variable VLLM_PLUGINS to control which plugins to load.
INFO 02-18 08:49:58 __init__.py:42] plugin ascend loaded.
INFO 02-18 08:49:58 __init__.py:174] Platform plugin ascend is activated
INFO 02-18 08:50:12 config.py:526] This model supports multiple tasks: {'embed', 'classify', 'generate', 'score', 'reward'}. Defaulting to 'generate'.
INFO 02-18 08:50:12 llm_engine.py:232] Initializing a V0 LLM engine (v0.7.1) with config: model='./Qwen2.5-0.5B-Instruct', speculative_config=None, tokenizer='./Qwen2.5-0.5B-Instruct', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=npu, decoding_config=DecodingConfig(guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=./Qwen2.5-0.5B-Instruct, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs=None, pooler_config=None, compilation_config={"splitting_ops":[],"compile_sizes":[],"cudagraph_capture_sizes":[256,248,240,232,224,216,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False,
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.86it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  5.85it/s]
INFO 02-18 08:50:24 executor_base.py:108] # CPU blocks: 35064, # CPU blocks: 2730
INFO 02-18 08:50:24 executor_base.py:113] Maximum concurrency for 32768 tokens per request: 136.97x
INFO 02-18 08:50:25 llm_engine.py:429] init engine (profile, create kv cache, warmup model) took 3.87 seconds
Processed prompts: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  8.46it/s, est. speed input: 46.55 toks/s, output: 135.41 toks/s]
Prompt: 'Hello, my name is', Generated text: " Shinji, a teenage boy from New York City. I'm a computer science"
Prompt: 'The president of the United States is', Generated text: ' a very important person. When he or she is elected, many people think that'
Prompt: 'The capital of France is', Generated text: ' Paris. The oldest part of the city is Saint-Germain-des-Pr'
Prompt: 'The future of AI is', Generated text: ' not bright\n\nThere is no doubt that the evolution of AI will have a huge'
```

##### 依赖三：AutoAlign及其依赖

```

pip install -e ".[ascend]"

````
