"""The rubric: the prompt the judge is given, and how its reply is read.

The rubric is the whole experiment. A judge is a language model asked to
apply a set of instructions, so the instructions have to say what they
mean and the reply has to be read defensively.

Three things in the prompt are doing real work.

The judge is asked to answer the question itself before it looks at the
answer under test, and to report that answer as ``my_answer``. That is what
makes the comparison a comparison. An earlier version of this rubric graded
the answer against the sources alone, and it graded the sources well: an
answer that repeated them was grounded. It also passed "Employees receive
18 annual leave days per calendar year" as the correct answer to "How many
sick leave days do employees receive?", while writing a reason that said it
"specifies annual leave days, not sick leave". A model can see a mismatch
and grade through it. Writing the answer first gives the mismatch something
to be measured against.

The reference answer is included, because it gives the judge a strong
anchor for factual correctness, and it is included with an explicit
instruction not to treat it as a wording target. The failure this guards
against is the one the token rule already has: an answer that is right but
says it differently is marked wrong. A judge that copies the reference
instead of reading the sources would reintroduce the same defect in a more
expensive form, and the validation against hand labels is what would catch
it.

The sources are numbered, so a reason can point at one, and so a reader
checking a verdict can find the passage it came from without guessing.

The judge is not told whether the question is answerable. The dataset
knows, but a judge that is told would be told the answer on the cases
where the answer is "there is no answer", which is the case that matters
most.

Only the four scores are requested. A separate boolean verdict would be a
second field that can contradict the first, and the binary verdict is
derived from the correctness score instead.
"""

from __future__ import annotations

import json
import re
from typing import Any

from rageval.judge.base import (
    DIMENSIONS,
    MAX_SCORE,
    MIN_SCORE,
    ORDER_ANSWER_FIRST,
    ORDER_REFERENCE_FIRST,
    RawJudgement,
)

SYSTEM_PROMPT = (
    "You are a strict grader of answers to questions about a company's "
    "knowledge base.\n"
    "You are given numbered SOURCES, a QUESTION, a REFERENCE ANSWER written "
    "by a human, and an ANSWER produced by another system.\n"
    "\n"
    "Grade the ANSWER as a response to the QUESTION.\n"
    "\n"
    "First, answer the QUESTION yourself, in one short phrase, and report "
    "it as my_answer. If a REFERENCE ANSWER is given, use it: it is a "
    "human's answer to this exact question. Otherwise answer from the "
    "SOURCES. If neither contains an answer, write exactly 'the sources do "
    "not say' and nothing more: say that the sources are silent, not that "
    "the thing is false.\n"
    "\n"
    "Then compare the ANSWER with your own answer:\n"
    "- The ANSWER is a correct response when it says the same thing as your "
    "answer, whether or not the wording matches.\n"
    "- The ANSWER is wrong when it says something else, even when every word "
    "in it is true and supported by the SOURCES. Being true is not a reason "
    "to pass an answer that answers a different question.\n"
    "\n"
    "Then score each dimension from 1 to 5:\n"
    "\n"
    "correctness  How right the ANSWER is as a response to the QUESTION, "
    "given the SOURCES.\n"
    "relevance    How directly the ANSWER addresses what was asked.\n"
    "groundedness How well every claim in the ANSWER is supported by the "
    "SOURCES.\n"
    "completeness How much of what the question asked for the ANSWER "
    "covers.\n"
    "\n"
    "Use the whole scale. 5 is exactly right, 4 is right but could be "
    "sharper, 3 is partly right, 2 is mostly wrong, 1 is wrong.\n"
    "\n"
    "Rules:\n"
    "1. Judge the ANSWER, not the REFERENCE. The REFERENCE is one human's "
    "wording for this question and may be longer or shorter than the "
    "ANSWER. An ANSWER that is factually right but phrased differently is "
    "not wrong for that reason.\n"
    "2. If the ANSWER declines to answer, look for an answer to this "
    "QUESTION in the SOURCES and in the REFERENCE ANSWER. If either of them "
    "contains one, the decline is wrong: the question had an answer and the "
    "system failed to produce it, so score correctness 1. If neither "
    "contains one, the decline is right: score all four dimensions 5.\n"
    "3. A claim the SOURCES do not support is ungrounded, however plausible "
    "it sounds.\n"
    "4. Reply with a single JSON object and nothing else:\n"
    '{"my_answer": "...", "correctness": 5, "relevance": 5, '
    '"groundedness": 5, "completeness": 5, "reason": "one short sentence"}'
)


def _numbered_sources(context: str) -> str:
    """Number the context passages so a reason can point at one."""
    passages = [line.strip() for line in context.splitlines() if line.strip()]
    if not passages:
        return "(no sources were retrieved)"
    return "\n".join(
        f"[{index}] {passage}" for index, passage in enumerate(passages, 1)
    )


def build_user_message(
    question: str,
    answer: str,
    context: str = "",
    reference_answer: str = "",
    order: str = ORDER_REFERENCE_FIRST,
) -> str:
    """Build the user message for one grading request.

    ``order`` decides whether the reference answer or the graded answer
    comes first. Swapping it is how the framework detects a judge that is
    sensitive to position rather than to content.
    """
    if order not in (ORDER_REFERENCE_FIRST, ORDER_ANSWER_FIRST):
        raise ValueError(
            f"Unknown order '{order}'. Use one of: "
            f"{ORDER_REFERENCE_FIRST}, {ORDER_ANSWER_FIRST}."
        )

    reference_block = ""
    if reference_answer.strip():
        reference_block = (
            "REFERENCE ANSWER (written by a human; it may be worded "
            f"differently):\n{reference_answer.strip()}"
        )
    answer_block = f"ANSWER TO GRADE:\n{answer.strip()}"

    sections = [
        f"SOURCES:\n{_numbered_sources(context)}",
        f"QUESTION: {question.strip()}",
    ]

    if order == ORDER_REFERENCE_FIRST:
        if reference_block:
            sections.append(reference_block)
        sections.append(answer_block)
    else:
        sections.append(answer_block)
        if reference_block:
            sections.append(reference_block)

    return "\n\n".join(sections)


def build_messages(
    question: str,
    answer: str,
    context: str = "",
    reference_answer: str = "",
    order: str = ORDER_REFERENCE_FIRST,
) -> list[dict[str, str]]:
    """Build the chat messages for one grading request."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_message(
                question, answer, context, reference_answer, order
            ),
        },
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response.

    Small models wrap their JSON in prose, fence it in a code block, or
    trail a note after it, so a strict parse alone would throw away most
    of the replies.
    """
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]

    if stripped.startswith("```"):
        unfenced = stripped.strip("`").strip()
        if unfenced.lower().startswith("json"):
            unfenced = unfenced[4:].strip()
        candidates.append(unfenced)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _as_score(value: Any) -> int | None:
    """Coerce a score to an integer between 1 and 5, or give up.

    A score outside the range is clamped rather than rejected: a judge
    that says 6 for a perfect answer has made an obvious mistake about the
    scale, and the meaning of the answer is still clear.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(MIN_SCORE, min(MAX_SCORE, value))
    if isinstance(value, float):
        return max(MIN_SCORE, min(MAX_SCORE, int(round(value))))
    if isinstance(value, str):
        match = re.search(r"-?\d+", value)
        if match:
            return max(MIN_SCORE, min(MAX_SCORE, int(match.group())))
    return None


def parse_judgement(text: str, order: str = ORDER_REFERENCE_FIRST) -> RawJudgement | None:
    """Read one judge reply.

    Returns ``None`` when the reply cannot be read at all. A reply that
    cannot be read is a failed sample rather than a score of one: treating
    an unparsable reply as "incorrect" would quietly bias every metric
    that depends on the judge.
    """
    payload = extract_json_object(text)
    if payload is None:
        return None

    scores: dict[str, int] = {}
    for dimension in DIMENSIONS:
        score = _as_score(payload.get(dimension))
        if score is None:
            return None
        scores[dimension] = score

    reason = payload.get("reason")
    if not isinstance(reason, str):
        reason = ""

    return RawJudgement(
        scores=scores,
        reason=reason.strip(),
        order=order,
        raw_text=text,
    )
