import enum
import os
from importlib import import_module
from typing import Sequence

import yaml

from autoalign.eval.safety.safety_eval_utils.schemas import TaskBase


def load_evaluation_tasks(tasks: list[str], wildguard_model_path: str = "", toxigen_roberta_model_path: str = "", llama_guard_model_path: str = "") -> Sequence[TaskBase]:
    """
    Load the evaluators for the given tasks.
    Args:
        tasks: List of tasks to evaluate the model on.
            each task can be xxx or xxx:yyy where xxx is the task name and yyy is the task config file name.
            if yyy is not provided, it will be assumed as xxx/default.yaml.

    Returns:
        List of evaluators.
    """
    evaluation_tasks = []
    tasks_dir = os.path.dirname(os.path.abspath(__file__))
    for task in tasks:
        # split by ":" to get the task name and the config file name.
        # if the config file name is not provided, it will be assumed as task/default.yaml.
        task_name = task if ":" not in task else task.split(":")[0]
        config_file_name = task.split(":")[-1] if ":" in task else "default"
        task_config_file = os.path.join(tasks_dir, task_name, f"{config_file_name}.yaml")

        # import the task module and check if the config file exists.
        module = import_module(f"safety_eval_utils.tasks.{task_name}")
        if not os.path.exists(task_config_file):
            raise FileNotFoundError(f"Task config file {task_config_file} does not exist.")

        # Load the task config file and create the task object
        with open(task_config_file, "r") as f:
            task_config = yaml.safe_load(f)
        
        task_class = getattr(module, task_config.pop("task_class"))
        if task_config["classifier_model_name"] == "WildGuard":
            task_config["local_model_path"] = wildguard_model_path
        elif task_config["classifier_model_name"] == "tomh/toxigen_roberta":
            task_config["local_model_path"] = toxigen_roberta_model_path
        elif task_config["classifier_model_name"] == "LlamaGuard3":
            task_config["local_model_path"] = llama_guard_model_path

        evaluation_tasks.append(task_class(**task_config))

    return evaluation_tasks
