CONFIG_CORE = """from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM, VLLM
with read_base():
    from ..opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from ..opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from ..opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from ..opencompass.configs.datasets.humaneval_cn.humaneval_cn_gen import humaneval_cn_datasets
    from ..opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from ..opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from ..opencompass.configs.summarizers.example import summarizer
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=10000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),),
)"""

CONFIG_ALL = """from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM, VLLM
with read_base():
    from ..opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from ..opencompass.configs.datasets.cmmlu.cmmlu_gen import cmmlu_datasets
    from ..opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
    from ..opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from ..opencompass.configs.datasets.math.math_gen import math_datasets
    from ..opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from ..opencompass.configs.datasets.humaneval_cn.humaneval_cn_gen import humaneval_cn_datasets
    from ..opencompass.configs.datasets.mbpp.mbpp_gen import mbpp_datasets
    from ..opencompass.configs.datasets.mbpp_cn.mbpp_cn_gen import mbpp_cn_datasets
    from ..opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets
    from ..opencompass.configs.datasets.bbh.bbh_gen import bbh_datasets
    from ..opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from ..opencompass.configs.summarizers.example import summarizer
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=20000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),),
)"""

CONFIG_HF = """
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask),),
)
datasets = sum([v for k, v in locals().items() if k.endswith("_datasets") or k == 'datasets'], [])
models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr="{model_name}",
        path="{model_path}",
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size={batch_size},
        run_cfg=dict(num_gpus={num_gpus}, num_procs=1),
        batch_padding=True,
    )
]"""

CONFIG_VLLM = """
eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=32,
        task=dict(type=OpenICLEvalTask),),
)
datasets = sum([v for k, v in locals().items() if k.endswith("_datasets") or k == 'datasets'], [])
models = [
    dict(type=VLLM,
    abbr="{model_name}",
    path="{model_path}",
    max_out_len=100,
    max_seq_len=2048,
    model_kwargs=dict(tensor_parallel_size=1,enforce_eager=True),
    generation_kwargs=dict(temperature=0),
    batch_size={batch_size},
    run_cfg=dict(num_gpus={num_gpus}, num_procs=1))
]"""
