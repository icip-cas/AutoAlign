import time

import fire

from autoalign.eval.safety.safety_eval_utils.models import load_generative_model
from autoalign.eval.safety.safety_eval_utils.schemas import OpenEndedTaskBase, VllmGeneratorModel
from autoalign.eval.safety.safety_eval_utils.tasks import load_evaluation_tasks
from autoalign.eval.safety.safety_eval_utils.utils import maybe_load_evaluation_report, save_evaluation_report


def generators(model_name_or_path: str,
               tasks: str,
               model_input_template_path_or_name: str,
               report_output_path: str,
               save_individual_results_path: str | None = None,
               override_existing_report: bool = False,
               use_vllm: bool = False,
               batch_size: int | None = None,
               filter_classifier_name: str | None = None,
               filter_model_override_path: str | None = None,
               tensor_parallel_size: int = 1,
               wildguard_model_path: str = "",
               toxigen_roberta_model_path: str = "",
               llama_guard_model_path: str = "",
               ):
    """
    Evaluate a model using command line arguments.
    Evaluation report will be saved into the json format, with the keys
    as the task names and the values as the dictionary of evaluation results per model.

    Example usage:
        python eval.py generators \
            --model_name_or_path ./finetuned_olmo_7b \
            --model_input_template_path_or_name olmo \
            --tasks harmbench \
            --report_output_path ./evaluation_report_olmo_generation.json

    Args:
        model_name_or_path: Name or path of the model to evaluate.
            If it is a name, it will be loading a model from the Hugging Face model hub, or using the API-based models (e.g. GPT-4).
            If it is a path, it will be loaded from the local path.

        tasks: List of tasks to evaluate the model on, split by comma.
            each task can be xxx or xxx:yyy where xxx is the task name and yyy is the task config file name.
            if yyy is not provided, it will be assumed as xxx/default.yaml.
            Then the yaml file for the task config will be loaded from tasks/generation/xxx/yyy.yaml.

        model_input_template_path_or_name: Path to the .txt file, which contains the chat template for the model.
            If it is a name, you can load from src/templates.
            If it is a file, the file should contain the input template for the model, which will be used to generate the responses for the evaluation tasks.
            For example, if the task is a single-turn dialogue task, it should contain {instruction} in its template.

        filter_classifier_name: Name of the classifier to filter out harmful prompts.
            If it is provided, all the prompts will be classified, and harmful prompts will be filtered out using the classifier.
            Harmful prompts will be answered by a canned refusal response.

        filter_model_override_path: Path to load a local model for the filter classifier.

        report_output_path: Path to save the evaluation report.

        save_individual_results_path: Path to save the individual results.

        override_existing_report: Whether to override the existing report or not. If true, it will override the existing report.

        use_vllm: Whether to use VLLM for generation or not.

        batch_size: Batch size for generation.
            If you use VLLM, batch_size will be automatically decided by the model.
    """
    start_time = time.time()
    if isinstance(tasks, tuple):
        tasks = list(tasks)
    elif isinstance(tasks, str):
        tasks = tasks.split(",")

    model = load_generative_model(
        model_name_or_path,
        model_input_template_path_or_name,
        use_vllm,
        batch_size,
        filter_classifier_name=filter_classifier_name,
        filter_model_override_path=filter_model_override_path,
        tensor_parallel_size=tensor_parallel_size,
    )
    evaluation_tasks: list[OpenEndedTaskBase] = load_evaluation_tasks(tasks, wildguard_model_path, toxigen_roberta_model_path, llama_guard_model_path)
    evaluation_report = maybe_load_evaluation_report(report_output_path)
    all_individual_results = maybe_load_evaluation_report(save_individual_results_path)

    for task_name, eval_task in zip(tasks, evaluation_tasks):
        if task_name in evaluation_report and not override_existing_report:
            print(f"Skipping {task_name} as it is already evaluated.")
            continue

        eval_task.validate_if_model_is_compatible(model)
        report, individual_results = eval_task.evaluate(model)
        evaluation_report[task_name] = report
        all_individual_results[task_name] = individual_results

    save_evaluation_report(report_output_path, evaluation_report)
    if save_individual_results_path:
        save_evaluation_report(save_individual_results_path, all_individual_results)

    if isinstance(model, VllmGeneratorModel) and model.delete_model_after_job_finished:
        model.delete_model_from_file_system()

    print(f" *** DONE! Evaluation report is saved to {report_output_path}. "
          f"Spent {time.time() - start_time:.2f} seconds.")


ALL_FUNCTIONS = {
    "generators": generators,
}


if __name__ == "__main__":
    fire.Fire(ALL_FUNCTIONS)
