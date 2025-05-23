import os

from autoalign.eval.safety.utils.schemas import (
    GeneratorModelBase,
    VllmGeneratorModel,
    HfGeneratorModel,
    APIGeneratorModel,
)


def load_generative_model(
    model_name_or_path: str,
    model_input_template_path_or_name: str,
    use_vllm: bool,
    batch_size: int | None = None,
    filter_classifier_name: str | None = None,
    filter_model_override_path: str | None = None,
    tensor_parallel_size: int = 1,
) -> GeneratorModelBase:
    """
    Load the model from the given name or path.
    Args:
        model_name_or_path: Name or path of the model to load.
            If it is a name, it will be loaded from the Hugging Face model hub, or using the API-based models (e.g. GPT-4).
                For an API model, the name should be structured as "<provider>:<identifier>", such as "openai:gpt-3.5-turbo".
            If it is a path, it will be loaded from the local path.

        model_input_template_path_or_name: Path to the .txt file, which contains the chat template for the model.
            If it is a name, you can load from src/templates.
            If it is a file, the file should contain the input template for the model, which will be used to generate the responses for the evaluation tasks.
            For example, if the task is a single-turn dialogue task, it should contain {instruction} in its template.

        use_vllm: Whether to use VLLM for generation or not.
    Returns:
        Model object.
    """
    if use_vllm:
        # Load the VLLM model
        delete_model_after_job_finished = False
        model = VllmGeneratorModel(
            model_name_or_path,
            model_input_template_path_or_name,
            delete_model_after_job_finished,
            filter_classifier_name=filter_classifier_name,
            filter_model_override_path=filter_model_override_path,
            tensor_parallel_size=tensor_parallel_size
        )
    else:
        if (
            "/" in model_name_or_path and len(model_name_or_path.split("/")) == 2
        ) or os.path.exists(model_name_or_path):
            # Load the HF model
            model = HfGeneratorModel(
                model_name_or_path=model_name_or_path,
                model_input_template_path_or_name=model_input_template_path_or_name,
                batch_size=batch_size,
                filter_classifier_name=filter_classifier_name,
                filter_model_override_path=filter_model_override_path,
            )
        else:
            # Load the API-based model
            return APIGeneratorModel(
                model_name_or_path=model_name_or_path,
                model_input_template_path_or_name=model_input_template_path_or_name,  # Unused by API model
                batch_size=batch_size,
                filter_classifier_name=filter_classifier_name,
                filter_model_override_path=filter_model_override_path,
            )

    return model
