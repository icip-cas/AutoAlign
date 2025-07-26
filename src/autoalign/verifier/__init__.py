from .xverify import (
    xverfiy_reward
)

from typing import Callable

def get_reward_funcs(script_args) -> list[Callable]:
    
    REWARD_FUNCS_REGISTRY = {
        "xverfiy_reward": xverfiy_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs