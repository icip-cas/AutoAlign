DEFAULT_TASK_GENERATION_PROMPT = """
Come up with a series of tasks and questions. Only the task/question, no further text/explanation, no additional information.
The task or question should be something a person would ask a chatbot.

"""
DEFAULT_TASK_PROMPT = "<task>{item}</task>\n"
# constants
# llm-as-judge prompt
# https://openreview.net/forum?id=uccHPGDlao

DEFAULT_REWARD_REGEX_TEMPLATE = """Score: {{ reward }}"""

from jinja2 import meta
import jinja2

jinja2_env = jinja2.Environment()


def find_variables_from_jinja_template(template: str):
    ast = jinja2_env.parse(template)
    return meta.find_undeclared_variables(ast)


def create_parse_reward_fn(reward_regex_template):
    assert find_variables_from_jinja_template(reward_regex_template) == {
        "reward"
    }, 'reward template must include "score" variable'
    reward_regex_str = jinja2_env.from_string(reward_regex_template).render(
        reward="([-+]?[0-9]+\.?[0-9]*)"
    )
    return reward_regex_str
