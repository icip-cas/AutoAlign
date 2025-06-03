import abc
import gc
from abc import ABC
import os

import torch
import json
import pandas as pd

from autoalign.eval.safety.utils.schemas import OpenEndedTaskBase, GeneratorModelBase
from autoalign.eval.safety.utils.classifier_models.base import SafetyClassifierBase, ResponseHarmfulness
from autoalign.eval.safety.utils.classifier_models.loader import load_classifier_model
from autoalign.eval.safety.utils.tasks.metrics import get_average


class CoCoNot(OpenEndedTaskBase, ABC):
    def __init__(self, classifier_model_name: str, local_model_path: str | None = None):
        super().__init__()
        self.max_new_tokens, self.top_p, self.repetition_penalty = self.prepare_hparams()
        self.classifier_model_name = classifier_model_name
        self.local_classifier_model_path = local_model_path

    def prepare_evaluator(self) -> SafetyClassifierBase:
        classifier: SafetyClassifierBase = load_classifier_model(self.classifier_model_name,
                                                                 local_model_path=self.local_classifier_model_path)
        assert classifier.get_required_input_fields() == ["prompt", "response"] or \
               classifier.get_required_input_fields() + classifier.get_optional_input_fields() == ["prompt",
                                                                                                   "response"], \
            "The classifier model must have input fields 'prompt' and 'response'."
        return classifier

    def _evaluate(self, model: GeneratorModelBase) -> tuple[dict, list[dict]]:
        completions = self.run_generation(model)
        return self.task_specific_eval(completions)

    def run_generation(self, model: GeneratorModelBase) -> list[dict]:
        """
        Runs a specific evaluation task based on provided task name.
        """
        original_data = self.data
        inputs = [{"instruction": ex["prompt"]} for ex in original_data]
        completions = model.generate_completions(
            inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Add completions to "res" column of data
        for ex, completion in zip(original_data, completions):
            ex["response"] = completion
        return original_data

    def task_specific_eval(self, completions: list[dict]) -> tuple[dict, list[dict]]:
        classifier = self.prepare_evaluator()

        res_data = classifier.classify(completions)
        for item, cls_result in zip(completions, res_data):
            item.update(cls_result.asdict())
        harmful_response_rate = []
        for json_data in completions:
            harmful_response_rate.append(json_data["response_harmfulness"] == ResponseHarmfulness.HARMFUL.value)

        del classifier
        torch.cuda.empty_cache()
        gc.collect()

        return {
            "harmful response rate (lower is better)": get_average(harmful_response_rate),
        }, completions

    @property
    def temperature(self):
        return 1.0

    def required_input_fields(self) -> list[str]:
        """
        Get the required input fields for the task.

        Returns:
            List of required input fields.
        """
        return ["instruction"]

    def prepare_hparams(self):
        return 512, 1.0, 1.0

    def load(self) -> list[dict]:
        """
        Load the task data. It will load data to self.data.

        Returns:
            List of task data.
        """
        data = []
        current_dir = "./data/eval/safety_eval/coconot"
        with open(os.path.join(current_dir, "coconot_eval.jsonl"), "r", encoding='utf-8') as f:
            for line in f.readlines():
                json_data = json.loads(line.strip())
                json_data = {"prompt": json_data["prompt"], "refusal_type": "origin"}
                data.append(json_data)
        with open(os.path.join(current_dir, "coconot_contrast_eval.jsonl"), "r", encoding='utf-8') as f:
            for line in f.readlines():
                json_data = json.loads(line.strip())
                json_data = {"prompt": json_data["prompt"], "refusal_type": "contrast"}
                data.append(json_data)
        return data