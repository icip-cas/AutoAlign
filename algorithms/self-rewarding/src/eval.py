import os
import subprocess
import argparse
import signal
from scipy.stats import spearmanr, kendalltau
from utils import load_json
from dpo_dataset_generator import parse_reward_fn
from templates import (
    DEFAULT_REWARD_REGEX_TEMPLATE,
    create_parse_reward_fn,
)
from autoalign.eval.default_configs import CONFIG_HF
from autoalign.conversation import Role, TEMPLATES
from autoalign.eval.default_configs import (
    META_TEMPLATE,
)
from autoalign.inference.inferencer import MultiProcessVllmInferencer

CONFIG_DATASETS = """from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM, VLLM
with read_base():
    from ....opencompass.configs.datasets.ARC_e.ARC_e_ppl import ARC_e_datasets
    from ....opencompass.configs.datasets.ARC_c.ARC_c_ppl import ARC_c_datasets
    from ....opencompass.configs.datasets.hellaswag.hellaswag_ppl import hellaswag_datasets
    from ....opencompass.configs.datasets.siqa.siqa_ppl import siqa_datasets
    from ....opencompass.configs.datasets.piqa.piqa_ppl import piqa_datasets
    from ....opencompass.configs.summarizers.example import summarizer
    from ....opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from ....opencompass.configs.datasets.mmlu.mmlu_ppl import mmlu_datasets
    from ....opencompass.configs.datasets.obqa.obqa_ppl import obqa_datasets
    from ....opencompass.configs.datasets.IFEval.IFEval_gen import ifeval_datasets
    from ....opencompass.configs.datasets.nq.nq_gen import nq_datasets
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

CONFIG_EVAL = """# 评测的模型标识名称
# The identifying name of the model to evaluate
model_name: {model_name}
# 评测时使用的上下文模板，可见src/autoalign/conversation.py中的TEMPLATES
template_name: {template_name}
# 评测的模型路径
# The path of the model to evaluate
model_path: {model_path}

# 评测的类型
# The type of evaluation
# 可选项：
# objective_core: 评测模型的核心客观指标。(evaluating the core objective metrics of the model)
# objective_all: 评测模型的所有客观指标。(evaluating all the objective metrics of the model)
# subjective: 评测模型的主观指标。(evaluating the subjective metrics of the model)
eval_type: subjective

# 单个模型 worker 所占用的GPU数量
# The number of GPUs occupied by a single model worker
per_model_gpu: {per_model_gpu}

# 单个 worker 的 batch_size
# The batch size of a single worker
batch_size: {batch_size}

# opencompass文件夹的路径
# The path of opencompass
opencompass_path: opencompass

# opencompass 的推理 backend
# The inference backend of opencompass
backend: {backend}

# ==============MTbench 设置================
# mtbench文件夹的路径
mt_path: data/eval/mt-bench

# ==============AlpacaEval 设置================
# see https://github.com/tatsu-lab/alpaca_eval/blob/main/src/alpaca_eval/evaluators_configs/README.md
# Recommend using: chatgpt_fn or weighted_alpaca_eval_gpt4_turbo
# use weighted_alpaca_eval_gpt4_turbo if you want the high agreement with humans.
# use chatgpt_fn if you are on a tight budget.
judge_model: chatgpt_fn

"""


def generate_config(
    model_name,
    model_path,
    per_model_gpu,
    batch_size,
    template_name,
):
    config = ""
    if not model_path.startswith("/"):
        model_path = os.path.join("../algorithms/self-rewarding/", model_path)
    template = ""
    config_path = "configs/{model_name}".format(model_name=model_name)
    if template_name != "none" and template_name != "None":
        config_path += "-with-template.py"
        if "llama-2" in template_name or "llama2" in template_name:
            template = META_TEMPLATE.format(
                human_begin="{!r}".format("<s>[INST] "),
                gpt_begin="{!r}".format(" "),
                human_end="{!r}".format(" [/INST]"),
                gpt_end="{!r}".format(" </s>"),
            )
        else:
            temp_obj = TEMPLATES[template_name]
            template = META_TEMPLATE.format(
                human_begin="{!r}".format(temp_obj.role_starts[Role.HUMAN]).replace(
                    "'", '"'
                ),
                gpt_begin="{!r}".format(temp_obj.role_starts[Role.ASSISTANT]).replace(
                    "'", '"'
                ),
                human_end="{!r}".format(temp_obj.role_ends[Role.HUMAN]).replace(
                    "'", '"'
                ),
                gpt_end="{!r}".format(temp_obj.role_ends[Role.ASSISTANT]).replace(
                    "'", '"'
                ),
            )
    else:
        config_path += "-no-template.py"

    config = CONFIG_DATASETS + CONFIG_HF.format(
        model_name=model_name,
        model_path=model_path,
        batch_size=batch_size,
        meta_template=template,
        num_gpus=per_model_gpu,
    )

    if not os.path.isdir("configs"):
        os.makedirs("configs")

    with open(config_path, "w") as file:
        file.write(config)
    return os.path.join("../algorithms/self-rewarding/", config_path)


def start_opencompass(work_dir, config_path, opencompass_path, reuse):
    command = [
        "cd {opencompass_path} \npython run.py {config_path} ".format(
            config_path=config_path, opencompass_path=opencompass_path
        )
    ]
    if reuse is not None:
        if not reuse.startswith("/"):
            reuse = os.path.join("../../../algorithms/self-rewarding/", reuse)
        command[0] += "--reuse {reuse_dir}".format(reuse_dir=reuse)
    else:
        command[0] += "--work-dir {work_dir}".format(
            work_dir=os.path.join("../algorithms/self-rewarding/", work_dir)
        )

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


def objective_eval(
    model_name,
    model_path,
    per_model_gpu,
    batch_size,
    opencompass_path,
    template_name,
    reuse,
):
    # config generation
    config_path = generate_config(
        model_name,
        model_path,
        per_model_gpu,
        batch_size,
        template_name,
    )
    work_dir = "outputs/{model_name}/opencompass_log/".format(model_name=model_name)
    # start_opencompass process
    start_opencompass(work_dir, config_path, opencompass_path, reuse)


def subjective_eval(
    model_name,
    model_path,
    per_model_gpu,
    batch_size,
    template_name,
    subjective_generate_only,
):
    if not os.path.isdir("configs"):
        os.makedirs("configs")
    config_path = "configs/{model_name}.yaml".format(model_name=model_name)
    model_path = os.path.join("algorithms/self-rewarding", model_path)
    config = CONFIG_EVAL.format(
        model_name=model_name,
        model_path=model_path,
        per_model_gpu=per_model_gpu,
        batch_size=batch_size,
        backend="hf",
        template_name=template_name,
    )
    with open(config_path, "w") as file:
        file.write(config)
    config_path = os.path.join("algorithms/self-rewarding", config_path)
    command = [
        "cd ../../ \n autoalign-cli eval --config-path {config_path}".format(
            config_path=config_path
        )
    ]
    if subjective_generate_only:
        command[0] += " --subjective_generate_only"
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
    return


def eval_llm_as_a_judge(model_path, per_model_gpu, eval_judge_file):
    inferencer = MultiProcessVllmInferencer(
        model_path=model_path, num_gpus_per_model=per_model_gpu
    )
    data = load_json(eval_judge_file)
    rewards_ground_truth = [
        int(float(eft_data["conversations"][1]["value"].replace("Score: ", "")))
        for eft_data in data
    ]
    responses = inferencer.inference(
        [eft_data["conversations"][0]["value"] for eft_data in data]
    )
    raw_scores = [
        parse_reward_fn(
            response,
            reward_regex_str=create_parse_reward_fn(DEFAULT_REWARD_REGEX_TEMPLATE),
        )
        for response in responses
    ]
    scores = []
    five_fullmark = 0
    exact_acc = 0
    for score, ground_truth in zip(raw_scores, rewards_ground_truth):
        if isinstance(score, int) or isinstance(score, float):
            if int(score) >= 0 and int(score) <= 5:
                scores.append(int(score))
                five_fullmark += 1
                if int(score) == int(ground_truth):
                    exact_acc += 1
                continue
        scores.append(-1)
    assert (
        len(data)
        == len(rewards_ground_truth)
        == len(responses)
        == len(raw_scores)
        == len(scores)
    )

    spearman, spearman_p = spearmanr(scores, rewards_ground_truth)
    kendall, kendall_p = kendalltau(scores, rewards_ground_truth)

    print("Five fullmark rate: {}%".format(round(five_fullmark / len(scores) * 100, 2)))
    print("Exact accuracy rate: {}%".format(round(exact_acc / len(scores) * 100, 2)))

    print("Spearman correlation: {}".format(round(spearman, 4)))
    print("Kendalltau correlation: {}".format(round(kendall, 4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        help="The chat template used in inference.",
    )
    parser.add_argument(
        "--reuse",
        type=str,
        required=False,
        default=None,
        help="The path of reused opencompass log folder.",
    )
    parser.add_argument(
        "--opencompass-path",
        type=str,
        required=False,
        help="The path of opencompass folder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The inference batch size.",
    )
    parser.add_argument(
        "--eval-type",
        type=str,
        choices=["objective", "subjective", "llm-as-a-judge"],
        help="The evaluation type, choices: objective, subjective and llm-as-a-judge",
    )
    parser.add_argument(
        "--eval-judge-data",
        type=str,
        help="The path of llm-as-a-judge evaluation data",
    )
    parser.add_argument("--subjective_generate_only", action="store_true")
    args = parser.parse_args()
    if args.eval_type == "objective":
        objective_eval(
            model_name=args.model_name,
            model_path=args.model_path,
            per_model_gpu=args.num_gpus_per_model,
            batch_size=args.batch_size,
            opencompass_path=args.opencompass_path,
            template_name=args.template_name,
            reuse=args.reuse,
        )
    elif args.eval_type == "subjective":
        subjective_eval(
            model_name=args.model_name,
            model_path=args.model_path,
            per_model_gpu=args.num_gpus_per_model,
            batch_size=args.batch_size,
            template_name=args.template_name,
            subjective_generate_only=args.subjective_generate_only,
        )
    elif args.eval_type == "llm-as-a-judge":
        eval_llm_as_a_judge(
            model_path=args.model_path,
            per_model_gpu=args.num_gpus_per_model,
            eval_judge_file=args.eval_judge_data,
        )
