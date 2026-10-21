"""The LLM-as-judge.

A judge grades one answer against a rubric, several times, in both orders,
and reports how much it agreed with itself. The parts are:

* ``base`` -- the item, the sample, the verdict, and the aggregation.
* ``rubric`` -- the prompt and the parsing of the reply.
* ``judge`` -- the rubric applied to a chat client, several times.
* ``openai_compat`` and ``ollama`` -- the two transports.
* ``cached`` -- a disk cache in front of either transport.
* ``agreement`` -- Cohen's kappa.
* ``labels`` -- the hand labels a judge is measured against.
"""

from rageval.judge.agreement import (
    agreement,
    cohen_kappa,
    interpret_kappa,
    weighted_kappa,
)
from rageval.judge.base import (
    DIMENSIONS,
    ORDER_ANSWER_FIRST,
    ORDER_REFERENCE_FIRST,
    ChatClient,
    JudgeError,
    JudgeItem,
    Judgement,
    RawJudgement,
    aggregate,
    lower_median,
    majority,
)
from rageval.judge.cached import CachedChatClient
from rageval.judge.judge import RubricJudge
from rageval.judge.labels import (
    CORRECT,
    INCORRECT,
    LabelledItem,
    load_labels,
    write_labels,
)
from rageval.judge.ollama import OllamaJudge
from rageval.judge.openai_compat import OpenAICompatJudge
from rageval.judge.rubric import build_messages, parse_judgement

__all__ = [
    "CORRECT",
    "DIMENSIONS",
    "INCORRECT",
    "ORDER_ANSWER_FIRST",
    "ORDER_REFERENCE_FIRST",
    "CachedChatClient",
    "ChatClient",
    "JudgeError",
    "JudgeItem",
    "Judgement",
    "LabelledItem",
    "OllamaJudge",
    "OpenAICompatJudge",
    "RawJudgement",
    "RubricJudge",
    "aggregate",
    "agreement",
    "build_messages",
    "cohen_kappa",
    "interpret_kappa",
    "load_labels",
    "lower_median",
    "majority",
    "parse_judgement",
    "weighted_kappa",
    "write_labels",
]
