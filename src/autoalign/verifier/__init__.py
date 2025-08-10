from .xverify import (
    xverify_reward
)

from typing import Callable

def get_reward_funcs(script_args) -> list[Callable]:
    
    REWARD_FUNCS_REGISTRY = {
        "xverify_reward": xverify_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs