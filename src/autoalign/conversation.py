from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
from transformers import AutoTokenizer
from abc import ABC, abstractmethod

IGNORED_TOKEN_ID = -100


class Role(Enum):
    SYSTEM = "system"
    HUMAN = "human"
    ASSISTANT = "gpt"


class RenderStrategy(ABC):
    @abstractmethod
    def get_conversation_str(
        self,
        messages: List[Tuple[Role, str]],
        template_attrs: Dict,
        add_generation_prompt: bool = False,
    ) -> str:
        pass

    @abstractmethod
    def generate_labels(
        self,
        messages: List[Tuple[Role, str]],
        template_attrs: Dict,
        tokenized_conversation,
        tokenizer,
    ) -> List[int]:
        pass


@dataclass
class ConversationTemplate:
    name: str
    role_starts: Optional[Dict[Role, str]] = None
    role_ends: Optional[Dict[Role, str]] = None
    offset: Optional[int] = 0
    default_system_message: Optional[str] = None
    strategy: Optional[RenderStrategy] = None
    stop_str: Optional[str] = None

    def get_attributes(self) -> Dict:
        return {
            "name": self.name,
            "role_starts": self.role_starts,
            "role_ends": self.role_ends,
            "offset": self.offset,
            "default_system_message": self.default_system_message,
        }


@dataclass
class Conversation:
    template: ConversationTemplate
    messages: List[Tuple[Role, str]] = field(default_factory=list)
    _system_message: Optional[str] = field(init=False, default=None)

    def __post_init__(self):
        self.system_message = self.template.default_system_message or ""

    def get_messages(self):
        return self.messages

    def _set_system_message(self, message: str):
        self._system_message = message
        if self.messages and self.messages[0][0] == Role.SYSTEM:
            self.messages[0] = (Role.SYSTEM, message)
        else:
            self.messages.insert(0, (Role.SYSTEM, message))

    def fill_in_messages(
        self, conv: Dict[str, Any], replace_conv_system_message: bool = True
    ):
        """Fill in conversation messages from an external source."""
        self.messages.clear()  # Clear existing messages

        # Handle system message
        first_message = conv["conversations"][0]
        if (
            first_message["from"] == Role.SYSTEM.value and replace_conv_system_message
        ):  # Use the system message from the conversation
            self._set_system_message(first_message["value"])
        else:
            self._set_system_message(
                self._system_message
            )  # Use the current system message

        # Fill in other messages
        for message_dict in conv["conversations"]:
            role_str, message_str = message_dict["from"], message_dict["value"]
            try:
                role = Role(role_str)
                if role != Role.SYSTEM:  # We've already handled the system message
                    self.messages.append((role, message_str))
            except ValueError:
                raise ValueError(
                    f"Invalid role: {role_str}. Must be one of {', '.join([r.value for r in Role])}"
                )

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""

        ret = []

        for msg in self.messages:
            role, content = msg
            if role == Role.SYSTEM and content:
                ret.append({"role": "system", "content": content})
            elif role == Role.HUMAN:
                ret.append({"role": "user", "content": content})
            elif role == Role.ASSISTANT:
                ret.append({"role": "assistant", "content": content})

        return ret

    @property
    def system_message(self) -> str:
        return self._system_message

    @system_message.setter
    def system_message(self, message: str):
        self._set_system_message(message)

    def append_message(self, role: Role, message: str):
        """Append a new message."""
        self.messages.append((role, message))

    def pop_message(self, index=-1):
        return self.messages.pop(index)

    def update_last_message(self, message: str):
        """Update the last message."""
        if self.messages:
            self.messages[-1] = (self.messages[-1][0], message)
        else:
            raise ValueError("No messages to update.")

    def clear_message(self):
        self.messages.clear()

    def get_conversation_str(self, add_generation_prompt: bool = False) -> str:
        """Get full conversation str"""
        if self.template.strategy:
            return self.template.strategy.get_conversation_str(
                self.messages, self.template.get_attributes(), add_generation_prompt
            )

        ret = ""
        for role, message in self.messages:
            ret += (
                self.template.role_starts[role]
                + message
                + self.template.role_ends[role]
            )
        if add_generation_prompt:
            ret += self.template.role_starts[Role.ASSISTANT]
        return ret

    def get_tokenized_conversation(
        self,
        tokenizer: AutoTokenizer,
        model_max_length: int,
        add_generation_prompt: bool = False,
    ):
        """Tokenize conversation and prepare labels"""
        conversation_str = self.get_conversation_str(
            add_generation_prompt=add_generation_prompt
        )
        tokenized_conversation = tokenizer(
            conversation_str, truncation=True, max_length=model_max_length
        )
        tokenized_conversation["labels"] = self._generate_labels(
            tokenized_conversation, tokenizer
        )

        return tokenized_conversation

    def _generate_labels(self, tokenized_conversation, tokenizer):
        if self.template.strategy:
            return self.template.strategy.generate_labels(
                self.messages,
                tokenized_conversation,
                tokenizer,
                self.template.get_attributes(),
            )

        labels = [IGNORED_TOKEN_ID] * len(tokenized_conversation.input_ids)
        cur_inst = ""
        for role, message in self.messages:
            if role in [Role.SYSTEM, Role.HUMAN]:
                cur_inst += (
                    self.template.role_starts[role]
                    + message
                    + self.template.role_ends[role]
                )
            else:
                cur_inst += self.template.role_starts[role]
                start_idx = len(tokenizer(cur_inst).input_ids) - self.template.offset
                end_idx = len(
                    tokenizer(
                        cur_inst + message + self.template.role_ends[role]
                    ).input_ids
                )
                labels[start_idx:end_idx] = tokenized_conversation.input_ids[
                    start_idx:end_idx
                ]
                cur_inst += message + self.template.role_ends[role]

        return labels

    def get_attributes(self) -> Dict:
        return {
            "template": self.template.get_attributes(),
            "system_message": self._system_message
            if self._system_message is not None
            else "",
            "role_ends": self.messages,
        }

    @classmethod
    def from_template(cls, template_name: str):
        """Get Conversation object from template_name"""
        template = TEMPLATES.get(template_name)
        if template is None:
            raise ValueError(f"Unknown conversation template: {template_name}")
        return cls(template=template)


class Llama2Strategy(RenderStrategy):

    """
    <s>[INST] <<SYS>>
    {{ system_prompt }}
    <</SYS>>

    {{ user_message_1 }} [/INST] {{ model_answer_1 }} </s>
    <s>[INST] {{ user_message_2 }} [/INST]

    If we use additional strategy, we no longer need to based on the template attributes
    """

    def get_conversation_str(
        self,
        messages: List[Tuple[Role, str]],
        template_attrs: Dict,
        add_generation_prompt: bool = False,
    ) -> str:
        ret = ""
        first_user_message = True
        system = False
        for role, message in messages:
            if role == Role.SYSTEM:
                ret += f"[INST] <<SYS>>\n{message}\n<</SYS>>"
                system = True
            elif role == Role.HUMAN:
                if first_user_message and system:
                    ret += f"\n\n{message} "
                else:
                    ret += f"<s>[INST] {message} "
                first_user_message = False
            elif role == Role.ASSISTANT:
                ret += f"[/INST] {message} </s>"
        if add_generation_prompt:
            ret += "[/INST]"
        return ret

    def generate_labels(
        self,
        messages: List[Tuple[Role, str]],
        tokenized_conversation,
        tokenizer,
        template_attrs: Dict,
    ) -> List[int]:
        labels = [IGNORED_TOKEN_ID] * len(tokenized_conversation.input_ids)
        cur_inst = ""
        first_user_message = True

        for role, message in messages:
            if role == Role.SYSTEM:
                cur_inst += f"[INST] <<SYS>>\n{message}\n<</SYS>>"
            elif role == Role.HUMAN:
                if first_user_message:
                    cur_inst += f"\n\n{message} [/INST]"
                    first_user_message = False
                else:
                    cur_inst += f"<s>[INST] {message} [/INST]"
            elif role == Role.ASSISTANT:
                start_idx = len(tokenizer(cur_inst).input_ids)
                cur_inst += f" {message} </s>"
                end_idx = len(tokenizer(cur_inst).input_ids)
                labels[start_idx:end_idx] = tokenized_conversation.input_ids[
                    start_idx:end_idx
                ]

        return labels


TEMPLATES = {
    "gpt-4": ConversationTemplate(
        name="gpt-4",
        default_system_message="",
    ),
    "vicuna_v1.1": ConversationTemplate(
        name="vicuna_v1.1",
        role_starts={
            Role.SYSTEM: "",
            Role.HUMAN: "USER: ",
            Role.ASSISTANT: "ASSISTANT: ",
        },
        role_ends={
            Role.SYSTEM: " ",
            Role.HUMAN: " ",
            Role.ASSISTANT: "</s>",
        },
        default_system_message="A chat between a curious user and an artificial intelligence assistant. \
            The assistant gives helpful, detailed, and polite answers to the user's questions.",
        offset=1,
        stop_str="</s>",
    ),
    "llama-2-chat": ConversationTemplate(
        name="llama-2-chat",
        default_system_message="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        offset=0,
        strategy=Llama2Strategy(),
        stop_str="</s>",
    ),
    "llama-2-chat-keep-system": ConversationTemplate(
        name="llama-2-chat-keep-system",
        offset=0,
        strategy=Llama2Strategy(),
        stop_str="</s>",
    ),
    "chatml": ConversationTemplate(
        name="chatml",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        default_system_message="You are a helpful assistant.",
        offset=0,
        stop_str="<|im_end|>",
    ),
    "chatml-keep-system": ConversationTemplate(
        name="chatml-keep-system",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        offset=0,
        stop_str="<|im_end|>",
    ),
    "llama-3-instruct": ConversationTemplate(
        name="llama-3-instruct",
        role_starts={
            Role.SYSTEM: "<|start_header_id|>system<|end_header_id|>\n\n",
            Role.HUMAN: "<|start_header_id|>user<|end_header_id|>\n\n",
            Role.ASSISTANT: "<|start_header_id|>assistant<|end_header_id|>\n\n",
        },
        role_ends={
            Role.SYSTEM: "<|eot_id|>",
            Role.HUMAN: "<|eot_id|>",
            Role.ASSISTANT: "<|eot_id|>",
        },
        offset=0,
        stop_str="<|eot_id|>",
    ),
    "mistral-instruct": ConversationTemplate(
        name="mistral-instruct",
        role_starts={
            Role.SYSTEM: "",
            Role.HUMAN: "[INST]",
            Role.ASSISTANT: "",
        },
        role_ends={
            Role.SYSTEM: "",
            Role.HUMAN: "[/INST]",
            Role.ASSISTANT: "</s>",
        },
        offset=0,
        stop_str="</s>",
    ),
    "gemma": ConversationTemplate(
        name="gemma",
        role_starts={
            Role.SYSTEM: "",
            Role.HUMAN: "<start_of_turn>user\n",
            Role.ASSISTANT: "<start_of_turn>model\n",
        },
        role_ends={
            Role.SYSTEM: "",
            Role.HUMAN: "<end_of_turn>\n",
            Role.ASSISTANT: "<end_of_turn>\n",
        },
        offset=0,
        stop_str="<end_of_turn>",
    ),
    "zephyr": ConversationTemplate(
        name="zephyr",
        role_starts={
            Role.SYSTEM: "<|system|>\n",
            Role.HUMAN: "<|user|>\n",
            Role.ASSISTANT: "<|assistant|>\n",
        },
        role_ends={
            Role.SYSTEM: "</s>",
            Role.HUMAN: "</s>",
            Role.ASSISTANT: "</s>",
        },
        offset=0,
        stop_str="</s>",
    ),
    "chatml-idsys": ConversationTemplate(
        name="chatml-idsys",
        role_starts={
            Role.SYSTEM: "<|im_start|>system\n",
            Role.HUMAN: "<|im_start|>user\n",
            Role.ASSISTANT: "<|im_start|>assistant\n",
        },
        role_ends={
            Role.SYSTEM: "<|im_end|>\n",
            Role.HUMAN: "<|im_end|>\n",
            Role.ASSISTANT: "<|im_end|>\n",
        },
        default_system_message="You are Zhuque, a conversational AI assistant trained by \
            Chinese Information Processing Laboratory (CIP). 你是朱雀，一个由中文信息处理实验室训练的对话式人工智能助手。\
            You are to give helpful, detailed, and polite answers to the user's questions. \
            你应当为用户的问题提供有帮助的、详细的、礼貌的回答。",
        offset=0,
        stop_str="<|im_end|>",
    ),
    "glm-4-chat": ConversationTemplate(
        # Not fully implemented
        name="glm-4-chat",
        role_starts={
            Role.SYSTEM: "<|system|>\n",
            Role.HUMAN: "<|user|>\n",
            Role.ASSISTANT: "<|assistant|>\n",
        },
        role_ends={
            Role.SYSTEM: "",
            Role.HUMAN: "",
            Role.ASSISTANT: "",
        },
        default_system_message="你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\
        保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。",
        offset=0,
        stop_str="<|user|>",
    ),
    "glm-4-chat-keep-system": ConversationTemplate(
        # Not fully implemented
        name="glm-4-chat",
        role_starts={
            Role.SYSTEM: "<|system|>\n",
            Role.HUMAN: "<|user|>\n",
            Role.ASSISTANT: "<|assistant|>\n",
        },
        role_ends={
            Role.SYSTEM: "",
            Role.HUMAN: "",
            Role.ASSISTANT: "",
        },
        offset=0,
        stop_str="<|user|>",
    ),
    "deepseek-coder-v1.5": ConversationTemplate(
        name="deepseek-coder-v1.5",
        role_starts={
            Role.SYSTEM: "",
            Role.HUMAN: "### Instruction:\n",
            Role.ASSISTANT: "### Response:\n",
        },
        role_ends={
            Role.SYSTEM: "",
            Role.HUMAN: "\n",
            Role.ASSISTANT: "\n<|EOT|>\n",
        },
        default_system_message="You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, \
and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, \
and other non-computer science questions, you will refuse to answer\n",
        offset=0,
        stop_str="<|EOT|>",
    ),
}

