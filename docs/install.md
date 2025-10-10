# Installation Guide
### Prerequisites
- Python 3.10
- Conda (Anaconda or Miniconda)
- CUDA-compatible GPU (for flash-attn support)

### Basic Installation

Create a new conda environment and install the base package:

```bash
conda create -n ata python=3.10
conda activate ata
pip install .
pip install .[flash-attn]
```

### Optional Components

#### 📊 Evaluation Environment

For running evaluations and benchmarks:

```bash
conda create -n ata_eval --clone ata
conda activate ata_eval
pip install .[eval]
bash ./scripts/post_install.sh
```

#### 🚀 Megatron Support

For distributed training with Megatron-LM:

```bash
conda activate ata
bash scripts/train/megatron/env_install.sh
```

> **Note:** Some version conflicts may occur during Megatron installation. These can typically be safely ignored if the installation completes successfully.

#### Ascend NPU Support
For users with **Ascend NPU devices**:
```bash
conda activate ata
pip install .
pip install .[train_ascend]
```


### Verification

After installation, verify your setup:

```bash
# Check if the package is installed
python -c "import autoalign; print('AutoAlign successfully installed!')"

# Or check with pip
pip show autoalign
```
#### Ascend verification

After `Ascend NPU Support` installation, verify your setup:

After you have completed the installation of the above dependencies, you can use the Python script below to verify the availability of `torch-npu`. The expected result is `True`.

```

import torch

import torch_npu

print(torch.npu.is_available())

```
To verify if `vLLM` has been successfully installed, you can use the following Python script.
```
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
You can then use the following command:
```
# Try `export VLLM_USE_MODELSCOPE=true` and `pip install modelscope`
# to speed up download if huggingface is not reachable.
python example.py
```
The expected result should be as follows:
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