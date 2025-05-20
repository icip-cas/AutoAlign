from autoalign.eval.safety.utils.classifier_models.base import SafetyClassifierBase, ConversationTurn, Role
from autoalign.eval.safety.utils.classifier_models.wildguard import WildGuard
from autoalign.eval.safety.utils.classifier_models.llama_guard_3 import LlamaGuard3


def load_classifier_model(model_name: str, **kwargs) -> SafetyClassifierBase:
    if model_name == "WildGuard":
        return WildGuard(**kwargs)
    elif model_name == "LlamaGuard3":
        return LlamaGuard3(**kwargs)
    else:
        raise ValueError(f"Model {model_name} not found.")
