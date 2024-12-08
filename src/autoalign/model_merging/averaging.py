import torch
from collections import defaultdict
from autoalign.model_merging.utils import get_param_names_to_merge


def average_merging(
    models_to_merge: list,
    average_weight: list = None,
    exclude_param_names_regex: list = [],
):
    """
    average merging method
    :param models_to_merge: list, individual models that need to be merged
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return: averaged parameters dictionary
    """
    if not average_weight:
        average_weight = [1 for _ in models_to_merge]
    else:
        assert len(models_to_merge) == len(
            average_weight
        ), "Merge weight should share the same number with the models!"
    # dictionary of list, where key is the parameter name,
    # value is a list of the corresponding parameters of all the models that need to be merged
    models_to_merge_param_dict = defaultdict(list)
    # iterate each individual model that needs to be merged

    for model_to_merge, weight in zip(models_to_merge, average_weight):
        param_dict = {
            param_name: param_value
            for param_name, param_value in model_to_merge.named_parameters()
        }
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(
            input_param_names=list(param_dict.keys()),
            exclude_param_names_regex=exclude_param_names_regex,
        )
        for param_name in param_names_to_merge:
            models_to_merge_param_dict[param_name].append(
                torch.multiply(weight, param_dict[param_name])
            )

    with torch.no_grad():
        # average merging of individual models' parameters
        averaged_params = {
            param_name: torch.div(
                torch.stack(model_to_merge_param, dim=0).sum(dim=0), sum(average_weight)
            )
            for param_name, model_to_merge_param in models_to_merge_param_dict.items()
        }
    return averaged_params
