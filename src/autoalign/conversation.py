"""conversation templates"""
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

from transformers import AutoTokenizer

# Ignored token id when calculating loss
IGNORED_TOKEN_ID = -100


@dataclass
class Conversation:
    """class to manage conversation template and data"""

    template_name: str
    role_starts: Dict[str, str]
    role_ends: Dict[str, str]
    overwrite_system_message: str = None
    default_system_message: str = None # when the data contains no system message and no overwrite_system_message is provided
    # roles: system, human, gpt
    # offset to handle non-additive tokenizers, i.e. tokenizer(s1+s2) != tokenizer(s1)+tokenizer(s2)
    # NB: whether tokenizer adds bos_token does not affect our label preparing
    # set to 1 if tokenizer adds an extra token when tokenizing assistant role_start alone
    # TODO: in theory, it is possible to automatically infer the offset given the tokenizer
    offset: int = 0
    # set to 1 if tokenizer adds a bos_token
    bos_offset: int = 0
    messages: List[Tuple[str]] = field(default_factory=list)

    def get_messages(self):
        return self.messages

    def get_conversation_str(self, add_generation_prompt=False):
        """get full conversation str"""
        ret = ""
        for role, message in self.messages:
            ret += self.role_starts[role] + message + self.role_ends[role]

        if add_generation_prompt:
            ret += self.role_starts["gpt"]

        return ret

    def fill_in_messages(self, conv):
        """fill in conversation messages"""
        # handle system_message
        if conv["conversations"][0]["from"] != "system":
            if self.overwrite_system_message is not None:
                self.messages.append(("system", self.overwrite_system_message))
            elif self.default_system_message is not None:
                self.messages.append(("system", self.default_system_message))
            else:
                self.messages.append(("system", ""))
        else:
            if self.overwrite_system_message is not None:
                self.messages.append(("system", self.overwrite_system_message))
            else:
                self.messages.append(("system", conv["conversations"][0]["value"]))

        # fill in messages
        for message_dict in conv["conversations"]:
            role, message_str = message_dict["from"], message_dict["value"]
            if role in ["human", "gpt"]:
                self.messages.append((role, message_str))

    def set_system_message(self, system_message: str):
        """Set the system message."""
        if self.messages[0]:
            self.messages.append(("system", system_message))
        else:
            self.messages[0][1] = system_message

    def get_system_message(self):
        """return the system message."""
        return self.messages[0][1] if self.messages[0] else None

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append((role, message))

    def update_last_message(self, message: str):
        """Update the last message.
        """
        self.messages[-1][1] = message

    def get_tokenized_conversation(
        self,
        tokenizer: AutoTokenizer,
        model_max_length: int,
    ):
        """tokenize conversation and prepare labels"""
        # get and tokenize full conversation str
        conversation_str = self.get_conversation_str()
        tokenized_conversation = tokenizer(conversation_str)

        # prepare labels
        full_id_len = len(tokenized_conversation.input_ids)
        labels = [IGNORED_TOKEN_ID] * full_id_len
        cur_inst = ""
        for role, message in self.messages:
            if role in ["system", "human"]:
                cur_inst += self.role_starts[role] + message + self.role_ends[role]
            else:
                cur_inst += self.role_starts[role]
                start_idx = len(tokenizer(cur_inst).input_ids) - self.offset
                end_idx = len(tokenizer(cur_inst + message + self.role_ends[role]).input_ids)
                labels[start_idx:end_idx] = tokenized_conversation.input_ids[start_idx:end_idx]
                cur_inst += message + self.role_ends[role]

        tokenized_conversation["labels"] = labels

        # NB: manually truncate to model_max_length
        tokenized_conversation["input_ids"] = tokenized_conversation["input_ids"][:model_max_length]
        tokenized_conversation["attention_mask"] = tokenized_conversation["attention_mask"][:model_max_length]
        tokenized_conversation["labels"] = tokenized_conversation["labels"][:model_max_length]

        return tokenized_conversation

    @classmethod
    def from_template(cls, template_name, overwrite_system_message:str=None):
        """get Conversation object from template_name"""
        if template_name == "vicuna_v1.1":
            return cls(
                template_name=template_name,
                role_starts={
                    "system": "",
                    "human": "USER: ",
                    "gpt": "ASSTSTANT: ",
                },
                role_ends={
                    "system": " ",
                    "human": " ",
                    "gpt": "</s>",
                },
                default_system_message="A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.",
                overwrite_system_message=overwrite_system_message,
                offset=1,
                bos_offset=1,
            )
        elif template_name == "chatml":
            return cls(
                template_name=template_name,
                role_starts={
                    "system": "<|im_start|>system\n",
                    "human": "<|im_start|>user\n",
                    "gpt": "<|im_start|>assistant\n",
                },
                role_ends={
                    "system": "<|im_end|>\n",
                    "human": "<|im_end|>\n",
                    "gpt": "<|im_end|>\n",
                },
                default_system_message="You are a helpful assistant.",
                overwrite_system_message=overwrite_system_message,
                offset=0,
                bos_offset=0,
            )
        elif template_name == "chatml-keep-system":
            return cls(
                template_name=template_name,
                role_starts={
                    "system": "<|im_start|>system\n",
                    "human": "<|im_start|>user\n",
                    "gpt": "<|im_start|>assistant\n",
                },
                role_ends={
                    "system": "<|im_end|>\n",
                    "human": "<|im_end|>\n",
                    "gpt": "<|im_end|>\n",
                },
                overwrite_system_message=overwrite_system_message,
                offset=0,
                bos_offset=0,
            )
        elif template_name == "llama-3-instruct":
            return cls(
                template_name=template_name,
                role_starts={
                    "system": "<s><|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
                    "human": "<|start_header_id|>user<|end_header_id|>\n\n",
                    "gpt": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                },
                role_ends={
                    "system": "<|eot_id|>",
                    "human": "<|eot_id|>",
                    "gpt": "<|eot_id|>",
                },
                # FIXME: what is the system_message for llama-3-instruct?
                overwrite_system_message=overwrite_system_message,
                offset=0,
                bos_offset=0,
            )
        elif template_name == "mistral-instruct":
            return cls(
                template_name=template_name,
                role_starts={
                    "system": "",
                    "human": "[INST]",
                    "gpt": "",
                },
                role_ends={
                    "system": "",
                    "human": "[/INST]",
                    "gpt": "</s>",
                },
                overwrite_system_message=overwrite_system_message,
                offset=0,
                bos_offset=0,
            )
        elif template_name == "chatml-idsys":
            return cls(
                template_name=template_name,
                role_starts={
                    "system": "<|im_start|>system\n",
                    "human": "<|im_start|>user\n",
                    "gpt": "<|im_start|>assistant\n",
                },
                role_ends={
                    "system": "<|im_end|>\n",
                    "human": "<|im_end|>\n",
                    "gpt": "<|im_end|>\n",
                },
                default_system_message="You are Zhuque, a conversational AI assistant "
                "trained by Chinese Information Processing Laboratory (CIP). "
                "你是朱雀，一个由中文信息处理实验室训练的对话式人工智能助手。"
                "You are to give helpful, detailed, and polite answers to the user's questions."
                "你应当为用户的问题提供有帮助的、详细的、礼貌的回答。",
                overwrite_system_message=overwrite_system_message,
                offset=0,
                bos_offset=0,
            )
        raise NotImplementedError("Unknown conversation template.")