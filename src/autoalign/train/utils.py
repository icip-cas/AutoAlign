import bisect
from typing import List, Sequence, Tuple
from autoalign.conversation import Conversation
import json


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


def split_list(lst, k):
    n = len(lst)
    base = n // k
    remainder = n % k
    result = []
    start = 0
    for i in range(k):
        if i < remainder:
            sub_len = base + 1
        else:
            sub_len = base
        end = start + sub_len
        result.append(lst[start:end])
        start = end
    return result


def pack_data_points_by_length(
    index_numbers: List[Tuple[int, int]], capacity: int, max_size: int = -1
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
    for idx, cur_length in index_numbers:
        if cur_length + current_concatenated_length <= capacity and (
            max_size == -1 or len(current_list) < max_size
        ):
            current_concatenated_length += cur_length
            current_list.append(idx)
        else:  # current_list is done, create a new one
            if len(current_list) > 0:
                result.append(current_list)
            current_list = [idx]
            current_concatenated_length = cur_length

    if len(current_list) > 0:
        result.append(current_list)

    # assert to make sure no indices were missing
    assert sum([len(indices) for indices in result]) == len(index_numbers)
    return result


def search_for_fit(numbers: Sequence[int], capacity: int) -> int:
    r"""
    Finds the index of largest number that fits into the knapsack with the given capacity.
    """
    index = bisect.bisect(numbers, capacity)
    return max(-1, index - 1)


def greedy_knapsack(
    index_numbers: List[Tuple[int, int]], capacity: int, max_size: int = -1
) -> List[List[int]]:
    r"""
    An efficient greedy algorithm with binary search for the knapsack problem.
    """
    lengths = len(index_numbers)
    # sorted_indices = [idx for idx, _ in sorted(enumerate(numbers), key=lambda x: x[1])]
    index_numbers.sort(
        key=lambda x: x[1]
    )  # sort numbers in ascending order for binary search
    sorted_indices, numbers = zip(*index_numbers)
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while max_size <= 0 or len(current_knapsack) < max_size:
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

def architecture_identification():
    import platform
    import torch

    PLATFORM = "cpu"
    arch = platform.machine().lower()

    device = torch.device("cpu")  

    try:
        import torch_npu  
        PLATFORM = "npu"
        device = torch.device("npu")
    except ImportError:
        if torch.cuda.is_available():
            PLATFORM = "gpu"
            device = torch.device("cuda")
        else:
            PLATFORM = "cpu"
            device = torch.device("cpu")

    return device, PLATFORM

def load_json(data_path: str):
    """
    Load data from JSON or JSONL file by detecting the actual format.
    
    Args:
        data_path (str): Path to the JSON/JSONL file
        
    Returns:
        list: List of dictionaries containing the data
    """
    data = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        # First, try to load as a single JSON file
        try:
            f.seek(0)  # Reset file pointer
            loaded_data = json.load(f)
            # Handle both list of objects and single object
            if isinstance(loaded_data, list):
                data = loaded_data
            else:
                data = [loaded_data]
            return data
        except json.JSONDecodeError:
            # If that fails, try to load as JSONL (each line is a JSON object)
            pass
    
    # Try JSONL format
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
        
        if not data:
            raise ValueError("No valid data found in file")
        
        return data
        
    except Exception as e:
        raise ValueError(f"Failed to parse file as either JSON or JSONL format. Error: {e}")
