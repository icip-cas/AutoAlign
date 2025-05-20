from .math_verifier import (
    accuracy_reward,
    format_reward,
    reasoning_steps_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    tag_count_reward
)

from .xverify import (
    xverfiy_reward
)

from typing import Callable

def get_reward_funcs(script_args) -> list[Callable]:
    
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "tag_count": tag_count_reward,
        "xverfiy_reward": xverfiy_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs