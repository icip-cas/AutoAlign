import bisect
from typing import List, Sequence
from autoalign.conversation import Conversation


def configure_model(conv_template_name, tokenizer, model):
    """specify eos token and bos token for model and tokenizer based on conversation template"""
    conversation = Conversation.from_template(conv_template_name)
    eos_token = conversation.template.stop_str
    eos_token_id = tokenizer(eos_token).input_ids[-1]
    # print(f"{tokenizer(eos_token)=} {eos_token=} {eos_token_id=} {tokenizer.decode([eos_token_id])=}")

    assert eos_token == tokenizer.decode(
        [eos_token_id]
    ), "eos token is not a valid token"
    tokenizer.eos_token_id = eos_token_id
    tokenizer.eos_token = eos_token

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.config.use_cache = False  # disable cache for training

    return None


def pack_data_points_by_length(
    lengths: List[int], max_length: int, max_size: int = -1
) -> List[List[int]]:
    """given lengths of data points, we merge consecutive data points into a new data point, as long as the concatenated length is less than max_length
    Args:
        lengths (List[int]): List of lengths of data points
        max_length (int): the concatenated length must be less than or equal max_length
        max_size: if != -1; the maximum number of consecutive items being merged; max_size: -1 --> no limit for number of items being merged

    max_size: the maximum number of data points being merged
    For example, lengths=[1, 3, 2, 2, 6, 4, 2, 6, 5]; max_length=10
    if max_size=-1 --> [[0,1,2,3], [4, 5], [6,7], [8]]
    if max_size=3 --> [[0,1,2], [3,4], [5, 6], [7], [8]]

    Returns:
        _type_: groups of indices: [[index1, index2, ...], [], ...]
    """
    result = []
    current_concatenated_length = 0
    current_list = []
    for i in range(len(lengths)):
        cur_length = lengths[i]
        if cur_length + current_concatenated_length <= max_length and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(i)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [i]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(lengths)
    return result


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return max(-1, index - 1)


def greedy_knapsack(
    numbers: List[int], capacity: int, max_size: int = -1
) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    lengths = len(numbers)
    sorted_indices = [idx for idx, _ in sorted(enumerate(numbers), key=lambda x: x[1])]
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while max_size < 0 or len(current_knapsack) < max_size:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers.pop(index)  # update the remaining capacity
            current_knapsack.append(
                sorted_indices.pop(index)
            )  # add the number to knapsack

        knapsacks.append(current_knapsack)

    assert sum([len(indices) for indices in knapsacks]) == lengths
    return knapsacks
