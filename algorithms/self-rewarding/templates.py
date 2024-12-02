import re
from utils import exists
from textwrap import dedent
from beartype.typing import Optional, Callable
from dataclasses import dataclass


DEFAULT_TASK_GENERATION_PROMPT = """
Come up with a series of tasks and questions. Only the task/question,
no further text/explanation, no additional information.
The task or question should be something a person would ask a chatbot.

"""
DEFAULT_TASK_PROMPT = "<task>{item}</task>\n"
# constants
# llm-as-judge prompt
# https://openreview.net/forum?id=uccHPGDlao

DEFAULT_LLM_AS_JUDGE_PROMPT = """
Review the user’s question and the corresponding response using the additive 5-point
scoring system described below. Points are accumulated based on the satisfaction of each
criterion:
- Add 1 point if the response is relevant and provides some information related to
the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question,
but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a
useful way, regardless of whether it seems to have been written by an AI Assistant or if it
has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective,
addressing the user’s question directly and comprehensively, and is well-organized and
helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question
by an AI Assistant, without extraneous information, reflecting expert knowledge, and
demonstrating a high-quality, engaging, and insightful answer.
User: {{ prompt }}
<response>{{ response }}</response>
After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”
Remember to assess from the AI Assistant perspective, utilizing web search knowledge as
necessary. To evaluate the response in alignment with this additive scoring model, we’ll
systematically attribute points based on the outlined criteria.
"""

DEFAULT_REWARD_REGEX_TEMPLATE = """Score: {{ reward }}"""

import jinja2

jinja2_env = jinja2.Environment()


def find_variables_from_jinja_template(template: str):
    from jinja2 import meta

    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {
        "reward"
    }, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(
        reward="([0-9\.]+)"
    )

    # @always(lambda: randrange(0, 10))
    def parse_reward_fn(llm_response: str) -> float:
        result = re.search(rf"{reward_regex_str}", llm_response)

        if not exists(result) or result.groups == 0:
            return None

        if not result.groups(1).isnumeric():
            return None

        return float(result.groups(1))

    return parse_reward_fn


# reward config


@dataclass
class RewardConfig:
    prompt_template: str
    reward_regex_template: Optional[str] = None
    parse_reward: Optional[Callable[[str], Optional[float]]] = None
    template_fn: Optional[Callable[..., str]] = None
    auto_dedent: bool = True

    def init(self):

        # maybe dedent

        if self.auto_dedent:
            self.prompt_template = dedent(self.prompt_template)

            if exists(self.reward_regex_template):
                self.reward_regex_template = dedent(self.reward_regex_template)

        # initialize render function for prompt and response template

        prompt_template = self.prompt_template
        assert find_variables_from_jinja_template(prompt_template) == {
            "prompt",
            "response",
        }, "template must include prompt and response templating variables"
        self.template_fn = jinja2_env.from_string(prompt_template).render

        # initialize the parse_reward if only the reward regex template is given

        if not exists(self.parse_reward):
            assert exists(
                self.reward_regex_template
            ), "reward_regex_template must be given if parse_reward is not passed in"
            self.parse_reward = create_parse_reward_fn(self.reward_regex_template)

        return self


# config, allowing for different types of reward prompting
# colocate with functions for extracting the response and reward

SELF_REWARD_PROMPT_CONFIG = dict(
    default=RewardConfig(
        prompt_template=DEFAULT_LLM_AS_JUDGE_PROMPT,
        reward_regex_template=DEFAULT_REWARD_REGEX_TEMPLATE,
    )
)


def default_is_valid_reward_pair(preferred_reward, unpreferred_reward):
    return (preferred_reward != unpreferred_reward).all()
