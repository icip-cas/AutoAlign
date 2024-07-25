import json
import subprocess
import time
import datasets
import yaml
import os
from transformers import AutoTokenizer
import tqdm
import pandas as pd
import signal
from argparse import ArgumentParser

from autoalign.inference.inferencer import MultiProcessVllmInferencer
from autoalign.utils import get_logger
from .inference_mt_bench import _run_mt_bench_eval, reorg_answer_file
from .gen_judgmen_mt_bench import judge


def parse_args(args: str):

    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, default=None, required=True)
    args = parser.parse_args(args)
    return args


logger = get_logger(__name__)


def generate_config(model_name, model_path, eval_type, per_model_gpu, batch_size, opencompass_path, backend):
    CONFIG_CORE = """from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM, VLLM
with read_base():
    from ..opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from ..opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from ..opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets
    from ..opencompass.configs.datasets.humaneval_cn.humaneval_cn_gen import humaneval_cn_datasets
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
    config = ""
    if not model_path.startswith("/"):
        model_path = os.path.join("..", model_path)

    if eval_type == "objective_core":
        if backend == "hf":
            config = CONFIG_CORE + CONFIG_HF.format(
                model_name=model_name, model_path=model_path, batch_size=batch_size, num_gpus=per_model_gpu
            )
        elif backend == "vllm":
            config = CONFIG_CORE + CONFIG_VLLM.format(
                model_name=model_name, model_path=model_path, batch_size=batch_size, num_gpus=per_model_gpu
            )
        else:
            raise ValueError("Error backend. Acceptable backend value: [hf, vllm]")
    elif eval_type == "objective_all":
        if backend == "hf":
            config = CONFIG_ALL + CONFIG_HF.format(
                model_name=model_name, model_path=model_path, batch_size=batch_size, num_gpus=per_model_gpu
            )
        elif backend == "vllm":
            config = CONFIG_ALL + CONFIG_VLLM.format(
                model_name=model_name, model_path=model_path, batch_size=batch_size, num_gpus=per_model_gpu
            )
        else:
            raise ValueError("Error backend. Acceptable backend value: [hf, vllm]")
    else:
        raise ValueError("Error eval_type. Acceptable eval_type value: [objective_core, objective_all]")
    if not os.path.isdir("configs"):
        os.makedirs("configs")
    config_path = "configs/{model_name}.py".format(model_name=model_name)
    with open(config_path, "w") as file:
        file.write(config)
    return os.path.join("../", config_path)


def start_opencompass(work_dir, config_path, opencompass_path):
    command = [
        "cd {opencompass_path} \n python run.py {config_path} --work-dir {work_dir}".format(
            config_path=config_path, opencompass_path=opencompass_path, work_dir=os.path.join("..", work_dir)
        )
    ]
    try:
        process = subprocess.Popen(command, shell=True, text=True, preexec_fn=os.setsid)
        # wait
        output, error = process.communicate()
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt: Ctrl+C, killing opencompass subprocess...")
        # try to kill subprocess
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        # handle other exceptions
    except Exception as e:
        print(f"Opencompass ended with errors: {e}")
    # print info
    if output is not None:
        print("Output:", output.decode("utf-8"))
    if error is not None:
        print("Error:", error.decode("utf-8"))
    # get return code of opencompass
    return_code = process.returncode
    print("Return code:", return_code)


def handle_result(model_name, work_dir):
    objective_datasets = [
        {"dataset": "gsm8k", "metric": "accuracy", "title": "GSM-8K(EN)"},
        {"dataset": "math", "metric": "accuracy", "title": "MATH(EN)"},
        {"dataset": "openai_humaneval", "metric": "humaneval_pass@1", "title": "HumanEval(EN)"},
        {"dataset": "openai_humaneval_cn", "metric": "humaneval_pass@1", "title": "HumanEval-CN(CH)"},
        {"dataset": "mbpp", "metric": "score", "title": "MBPP(EN)"},
        {"dataset": "mbpp_cn", "metric": "score", "title": "MBPP-CN(CH)"},
        {"dataset": "mmlu", "metric": "naive_average", "title": "MMLU(EN)"},
        {"dataset": "GPQA_diamond", "metric": "accuracy", "title": "GPQA(EN)"},
        {"dataset": "cmmlu", "metric": "naive_average", "title": "CMMLU(CH)"},
        {"dataset": "ceval", "metric": "naive_average", "title": "C-Eval(CH)"},
    ]
    ordered_df = pd.DataFrame(
        columns=[
            "MT-Bench(EN)",
            "AlpacaEval 1.0(EN)",
            "AlpacaEval 2.0(EN)",
            "AlignBench(CH)",
            "MATH(EN)",
            "GSM-8K(EN)",
            "SuperCLUE-Math6(CH)",
            "HumanEval(EN)",
            "MBPP(EN)",
            "HumanEval-CN(CH)",
            "MBPP-CN(CH)",
            "MMLU(EN)",
            "GPQA(EN)",
            "CMMLU(CH)",
            "C-Eval(CH)",
        ]
    )
    assert os.path.isdir(work_dir)
    encounter_model_name = False
    latest_file = None
    new_row = {}
    # Traverse all the first-level subdirectories under the 'outputs' folder of opencompass.
    for eval in os.listdir(work_dir):
        evaluation_path = os.path.join(work_dir, eval, "summary")
        # For a given evaluation, determine whether a 'summary' folder exists; ignore folders that do not have a 'summary.'
        if not os.path.isdir(evaluation_path):
            continue
        # Traverse from front to back
        for file in os.listdir(evaluation_path):
            if "csv" in file:
                # overwrite as you go
                file_name = os.path.join(evaluation_path, file)
                df = pd.read_csv(file_name)
                if model_name in df.columns:
                    if not encounter_model_name:
                        encounter_model_name = True
                        latest_file = file_name
                    else:
                        if latest_file.split("/")[-1] < file_name.split("/")[-1]:
                            print("Warning: Duplicate model_name: {}, latest raw_result file name: {}".format(model_name, file_name))
                            latest_file = file_name
                        else:
                            continue
                    new_row = {}
                    for dataset in objective_datasets:
                        for row in df.iterrows():
                            row = row[1]
                            if row["dataset"] == dataset["dataset"] and row["metric"] == dataset["metric"]:
                                new_row[dataset["title"]] = row[model_name]
    ordered_df.loc[len(ordered_df)] = new_row
    return ordered_df


def warn_duplicate(model_name, position):
    print("Found duplicate model_name: {} in {}. This may cause overwriting. Continue(y) or Exit(n)? [y/n]".format(model_name, position))
    while True:
        in_content = input()
        if in_content.lower() == "y":
            break
        elif in_content.lower() == "n":
            print("Process exit with code 1. Try to change your model_name")
            exit(1)
        else:
            print("Invalid input, tap your keyboard again.")


def build_data(datas, batch_size, tokenizer, inferencer: MultiProcessVllmInferencer) -> list:
    sorted_datas = sorted(datas, key=lambda x: len(x["instruction"]))
    dealdatas = []
    batch_num = (len(sorted_datas) - 1) // batch_size + 1
    for i in tqdm.tqdm(range(batch_num)):
        batch_datas = sorted_datas[i * batch_size : (i + 1) * batch_size]
        prompts = []
        for data in batch_datas:
            messages = [{"role": "user", "content": data["instruction"]}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        outputs = inferencer.inference(prompts)
        for j, data in enumerate(batch_datas):
            data["prompt"] = data["instruction"]
            data["output"] = outputs[j]
            dealdatas.append(data.copy())
    return dealdatas


def run_alpaca_eval(model_name: str, model_path: str, batch_size: int, inferencer: MultiProcessVllmInferencer) -> None:
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    logger.info(f"Running ALPaCA evaluation for model: {model_name}, model_path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("starting to generate outputs")
    try:
        datas = build_data(eval_set, batch_size=batch_size, tokenizer=tokenizer, inferencer=inferencer)
    except Exception as e:
        logger.error(f"error when generating alpaca outputs: {e}")
        return
    for data in datas:
        data["generator"] = model_name
    if not os.path.exists(f"data/alpaca/{model_name}"):
        os.makedirs(f"data/alpaca/{model_name}")
    with open(f"data/alpaca/{model_name}/{model_name}_outputs.json", "w") as f:
        json.dump(datas, f, indent=4)
    logger.info("outputs generated, starting to evaluate")
    command = f"""alpaca_eval --model_outputs data/alpaca/{model_name}/{model_name}_outputs.json \
    --output_path data/alpaca/{model_name}/ \
    --annotators_config weighted_alpaca_eval_gpt4_turbo \
    --is_overwrite_leaderboard"""
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        logger.error("alpaca evaluation failed, we will continue to run the next evaluation")
    else:
        logger.info("alpaca evaluation finished")
    return


def display_result_single(bench_name: str, input_file: str, judge_model: str, model_list: list) -> None:
    if input_file is None:
        input_file = f"data/{bench_name}/model_judgment/{judge_model}_single.jsonl"
    else:
        input_file = input_file

    print(f"Input file: {input_file}")
    df_all = pd.read_json(input_file, lines=True)
    df = df_all[["model", "score", "turn"]]
    df = df[df["score"] != -1]

    if model_list is not None:
        df = df[df["model"].isin(model_list)]

    print("\n########## First turn ##########")
    df_1 = df[df["turn"] == 1].groupby(["model", "turn"]).mean()
    print(df_1.sort_values(by="score", ascending=False))

    if bench_name == "mt_bench":
        print("\n########## Second turn ##########")
        df_2 = df[df["turn"] == 2].groupby(["model", "turn"]).mean()
        print(df_2.sort_values(by="score", ascending=False))

        print("\n########## Average ##########")
        df_3 = df[["model", "score"]].groupby(["model"]).mean()
        print(df_3.sort_values(by="score", ascending=False))


def run_mt_bench_eval(model_path: str, model_name: str, batch_size: int, mtpath: str, num_gpus_per_model: int, template_name: str) -> None:
    batch_size = max(8, batch_size)
    logger.info(f"Running MT-Bench evaluation for model: {model_name}, model_path: {model_path}")

    question_file = f"{mtpath}/question.jsonl"
    answer_file = f"{mtpath}/model_answer/{model_name}.jsonl"

    _run_mt_bench_eval(
        model_path=model_path,
        model_id=model_name,
        template_name=template_name,
        question_file=question_file,
        answer_file=answer_file,
        question_begin=None,
        question_end=None,
        max_new_token=1024,
        num_choices=1,
        num_gpus_per_model=num_gpus_per_model,
    )

    reorg_answer_file(answer_file)

    logger.info("mt-bench generation finished")
    logger.info("starting to evaluate")
    judge(model_list=[model_name], parallel=4, mtpath=mtpath)
    logger.info("mt-bench evaluation finished")

    display_result_single("mt_bench", None, "gpt-4", [model_name])


def transpose_and_format_dataframe(input_dataframe, output_txt_path):
    transposed_df = input_dataframe.transpose()
    max_index_len = pd.Series([len(str(x)) for x in transposed_df.index]).max()
    max_widths = transposed_df.applymap(lambda x: len(str(x))).max()
    max_widths = max_widths.tolist()
    max_widths.insert(0, max_index_len)
    with open(output_txt_path, "w", encoding="utf-8") as txt_file:
        for row in transposed_df.itertuples(index=True, name=None):
            formatted_row = "\t\t".join(
                str(val if not pd.isna(row[idx]) else "-").ljust(max_widths[idx]) for idx, val in enumerate(list(row))
            )
            txt_file.write(formatted_row + "\n")


def run_objective_eval(model_name, model_path, eval_type, per_model_gpu, batch_size, opencompass_path, backend):
    # check duplicate model_name
    if os.path.exists("configs/{model_name}.py".format(model_name=model_name)):
        warn_duplicate(model_name, "configs")
    if os.path.exists("outputs/{model_name}/ordered_res.csv".format(model_name=model_name)) or os.path.exists(
        "outputs/{model_name}/ordered_res.txt".format(model_name=model_name)
    ):
        warn_duplicate(model_name, "ordered_results")
    work_dir = "outputs/{model_name}/opencompass_log/".format(model_name=model_name)
    if os.path.isdir(work_dir):
        check_duplicate = False
        for eval in os.listdir(work_dir):
            if check_duplicate:
                break
            evaluation_path = os.path.join(work_dir, eval, "summary")
            if not os.path.isdir(evaluation_path):
                continue
            for file in os.listdir(evaluation_path):
                if "csv" in file:
                    file_name = os.path.join(evaluation_path, file)
                    df = pd.read_csv(file_name)
                    if model_name in df.columns:
                        warn_duplicate(model_name, "raw_results")
                        check_duplicate = True
                        break
    # config generation
    config_path = generate_config(model_name, model_path, eval_type, per_model_gpu, batch_size, opencompass_path, backend)
    # start_opencompass process
    start_opencompass(work_dir, config_path, opencompass_path)
    # summarize the results and display
    ordered_df = handle_result(model_name, work_dir)
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    ordered_df.to_csv("outputs/{model_name}/ordered_res.csv".format(model_name=model_name), index=False)
    transpose_and_format_dataframe(ordered_df, "outputs/{model_name}/ordered_res.txt".format(model_name=model_name))


def run_eval(args) -> None:
    # 若args中存在config_path
    args = parse_args(args)
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    model_name = config["model_name"]
    # 获取time_stamp拼接至model_name
    time_stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    model_name = f"{model_name}_{time_stamp}"
    model_path = config["model_path"]
    batch_size = config["batch_size"]
    eval_type = config["eval_type"]
    template_name = config["template_name"]
    # num_gpus = torch.cuda.device_count()
    mtpath = config["mt_path"]
    per_model_gpu = config["per_model_gpu"]
    opencompass_path = config["opencompass_path"]
    opencompass_backend = config["backend"]

    if eval_type.startswith("objective"):
        run_objective_eval(model_name, model_path, eval_type, per_model_gpu, batch_size, opencompass_path, opencompass_backend)
    elif eval_type == "subjective":
        # 测试是否已设置OPENAI_BASE_URL和OPENAI_API_KEY
        if not os.environ.get("OPENAI_BASE_URL") or not os.environ.get("OPENAI_API_KEY"):
            logger.error("OPENAI_BASE_URL or OPENAI_API_KEY not set")
        inferencer = MultiProcessVllmInferencer(model_path=model_path, num_gpus_per_model=per_model_gpu)
        run_alpaca_eval(model_name, model_path, batch_size, inferencer)
        run_mt_bench_eval(model_path, model_name, batch_size, mtpath, per_model_gpu, template_name)
    else:
        logger.error(f"eval_type {eval_type} not supported")
        return
