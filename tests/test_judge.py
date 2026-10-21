"""Tests for the LLM-as-judge.

None of these tests need a running model server. The HTTP layer is
replaced, so the suite still passes on a machine where neither Ollama nor
a hosted judge is available.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from rageval.config import EvalConfig
from rageval.dataset import load_dataset
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
    JudgeError,
    JudgeItem,
    RawJudgement,
    aggregate,
    majority,
)
from rageval.judge.cached import CachedChatClient
from rageval.judge.judge import RubricJudge
from rageval.judge.labels import LabelledItem, load_labels, write_labels
from rageval.judge.ollama import OllamaJudge
from rageval.judge.openai_compat import OpenAICompatJudge
from rageval.judge.rubric import (
    build_messages,
    extract_json_object,
    parse_judgement,
)
from rageval.judge.validation import token_rule_verdict, validate
from rageval.runner import build_judge, evaluate_case

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "data" / "hand_labels.json"


class FakeResponse:
    """Stands in for the object returned by urlopen."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def verdict(
    correctness: int = 5,
    relevance: int = 5,
    groundedness: int = 5,
    completeness: int = 5,
    reason: str = "because",
) -> str:
    """Build a judge reply in the shape the rubric asks for."""
    return json.dumps(
        {
            "my_answer": "Bengaluru",
            "correctness": correctness,
            "relevance": relevance,
            "groundedness": groundedness,
            "completeness": completeness,
            "reason": reason,
        }
    )


def capture_ollama(monkeypatch, content: str) -> dict[str, Any]:
    """Capture the request an Ollama judge makes and answer it."""
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["headers"] = request.headers
        seen["timeout"] = timeout
        return FakeResponse({"message": {"role": "assistant", "content": content}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def capture_openai(monkeypatch, content: str) -> dict[str, Any]:
    """Capture the request an OpenAI compatible judge makes and answer it."""
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["headers"] = request.headers
        seen["timeout"] = timeout
        return FakeResponse(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


class RecordingClient:
    """A chat client that answers from a script and records the calls."""

    name = "recording"

    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("no reply was scripted")
        return self.replies.pop(0)


# --- the rubric ------------------------------------------------------------


def test_the_sources_are_numbered():
    message = build_messages("Where?", "Bengaluru", "First line.\nSecond line.")[1][
        "content"
    ]

    assert "[1] First line." in message
    assert "[2] Second line." in message


def test_the_reference_comes_first_in_that_order():
    message = build_messages(
        "Where?",
        "Bengaluru",
        "The office is in Bengaluru.",
        "Bengaluru, Karnataka.",
        ORDER_REFERENCE_FIRST,
    )[1]["content"]

    assert message.index("REFERENCE ANSWER") < message.index("ANSWER TO GRADE")


def test_the_answer_comes_first_in_that_order():
    message = build_messages(
        "Where?",
        "Bengaluru",
        "The office is in Bengaluru.",
        "Bengaluru, Karnataka.",
        ORDER_ANSWER_FIRST,
    )[1]["content"]

    assert message.index("ANSWER TO GRADE") < message.index("REFERENCE ANSWER")


def test_the_two_orders_hold_the_same_text():
    """Swapping the order must not change what is asked, only where it sits."""
    reference_first = build_messages(
        "Where?", "Bengaluru", "context", "Bengaluru, Karnataka.", ORDER_REFERENCE_FIRST
    )[1]["content"]
    answer_first = build_messages(
        "Where?", "Bengaluru", "context", "Bengaluru, Karnataka.", ORDER_ANSWER_FIRST
    )[1]["content"]

    for block in ("Where?", "Bengaluru", "context", "Bengaluru, Karnataka."):
        assert block in reference_first
        assert block in answer_first


def test_an_empty_reference_is_left_out_entirely():
    """An unanswerable question has no reference, and saying so would be a hint."""
    message = build_messages("Where?", "I don't know.", "context", "")[1]["content"]

    assert "REFERENCE ANSWER" not in message
    assert "I don't know." in message


def test_an_unknown_order_is_rejected():
    with pytest.raises(ValueError, match="Unknown order"):
        build_messages("Where?", "Bengaluru", "", "", "sideways")


def test_the_system_prompt_asks_for_the_four_scores():
    system = build_messages("Where?", "Bengaluru")[0]["content"]

    for dimension in DIMENSIONS:
        assert dimension in system


# --- reading the reply -----------------------------------------------------


def test_it_reads_the_scores_out_of_a_reply():
    parsed = parse_judgement(verdict(4, 3, 2, 1))

    assert parsed is not None
    assert parsed.scores == {
        "correctness": 4,
        "relevance": 3,
        "groundedness": 2,
        "completeness": 1,
    }
    assert parsed.reason == "because"


def test_a_score_outside_the_scale_is_clamped():
    parsed = parse_judgement(verdict(correctness=9, groundedness=0))

    assert parsed is not None
    assert parsed.scores["correctness"] == 5
    assert parsed.scores["groundedness"] == 1


def test_a_score_written_as_a_string_is_read():
    reply = '{"correctness": "4", "relevance": 4, "groundedness": 4, "completeness": 4}'
    parsed = parse_judgement(reply)

    assert parsed is not None
    assert parsed.scores["correctness"] == 4


def test_it_finds_json_wrapped_in_prose():
    parsed = parse_judgement('Sure! {"correctness": 5, "relevance": 5, '
                             '"groundedness": 5, "completeness": 5} Hope that helps.')

    assert parsed is not None
    assert parsed.scores["correctness"] == 5


def test_it_finds_json_inside_a_code_fence():
    parsed = parse_judgement("```json\n" + verdict() + "\n```")

    assert parsed is not None
    assert parsed.scores["correctness"] == 5


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I would rather not say.",
        "{not json at all}",
        '{"correctness": 5, "relevance": 5}',
    ],
)
def test_a_reply_that_cannot_be_read_yields_nothing(text):
    """An unreadable reply is a failed sample, not a score of one."""
    assert parse_judgement(text) is None


def test_a_missing_reason_is_not_fatal():
    reply = '{"correctness": 5, "relevance": 5, "groundedness": 5, "completeness": 5}'
    parsed = parse_judgement(reply)

    assert parsed is not None
    assert parsed.reason == ""


def test_json_extraction_handles_the_shapes_models_produce():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('before {"a": 1} after') == {"a": 1}
    assert extract_json_object("no json") is None


# --- aggregation -----------------------------------------------------------


def test_the_median_is_used_so_one_wild_sample_cannot_move_the_result():
    raws = [
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 1 for dimension in DIMENSIONS}),
    ]

    judgement = aggregate(raws)

    assert judgement.correctness == 5
    assert judgement.correct is True


def test_the_verdict_is_a_majority_vote():
    raws = [
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 1 for dimension in DIMENSIONS}),
    ]

    judgement = aggregate(raws)

    assert judgement.correct is True
    assert judgement.sample_agreement == pytest.approx(2 / 3)


def test_the_threshold_decides_the_verdict():
    raws = [RawJudgement(scores={dimension: 3 for dimension in DIMENSIONS})]

    assert aggregate(raws, threshold=4).correct is False
    assert aggregate(raws, threshold=3).correct is True


def test_the_reason_comes_from_the_sample_that_matches_the_verdict():
    raws = [
        RawJudgement(scores={dimension: 1 for dimension in DIMENSIONS}, reason="no"),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}, reason="yes"),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}, reason="also"),
    ]

    assert aggregate(raws).reason == "yes"


def test_an_item_with_no_samples_is_an_error():
    with pytest.raises(JudgeError):
        aggregate([])


def test_majority_breaks_a_tie_against_correct():
    """An exactly undecided judge does not get to call the answer correct."""
    assert majority([True, False]) is False
    assert majority([True, True, False]) is True
    assert majority([False, False, True]) is False


def test_majority_can_be_told_how_to_break_a_tie():
    assert majority([True, False], ties_to=True) is True


def test_the_verdict_and_the_reported_score_never_contradict():
    """A reader has to be able to work the verdict out from the score.

    Six samples split three and three is the case that used to break:
    the average of the two middle samples said 4 and the verdict said
    "incorrect", so the report contradicted itself.
    """
    raws = [
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 3 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 3 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 3 for dimension in DIMENSIONS}),
    ]

    judgement = aggregate(raws, threshold=4)

    assert judgement.correctness == 3
    assert judgement.correct is False
    assert judgement.correct == (judgement.correctness >= 4)


def test_the_lower_median_is_used_rather_than_the_average():
    """The reported score has to be a score one of the samples gave."""
    raws = [
        RawJudgement(scores={dimension: 3 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 4 for dimension in DIMENSIONS}),
        RawJudgement(scores={dimension: 5 for dimension in DIMENSIONS}),
    ]

    assert aggregate(raws).correctness == 4


def test_the_verdict_follows_the_score_for_every_split():
    for yes in range(7):
        raws = [
            RawJudgement(
                scores={
                    dimension: (5 if index < yes else 3)
                    for dimension in DIMENSIONS
                }
            )
            for index in range(6)
        ]

        judgement = aggregate(raws, threshold=4)

        assert judgement.correct == (judgement.correctness >= 4)
        assert judgement.correct == (yes > 3)


def test_the_order_consistency_is_reported():
    raws = [
        RawJudgement(scores={d: 5 for d in DIMENSIONS}, order=ORDER_REFERENCE_FIRST),
        RawJudgement(scores={d: 5 for d in DIMENSIONS}, order=ORDER_ANSWER_FIRST),
    ]

    assert aggregate(raws).order_consistent is True

    raws = [
        RawJudgement(scores={d: 5 for d in DIMENSIONS}, order=ORDER_REFERENCE_FIRST),
        RawJudgement(scores={d: 1 for d in DIMENSIONS}, order=ORDER_ANSWER_FIRST),
    ]

    assert aggregate(raws).order_consistent is False


def test_order_consistency_is_none_when_the_order_was_not_swapped():
    raws = [RawJudgement(scores={d: 5 for d in DIMENSIONS})]

    assert aggregate(raws).order_consistent is None


# --- agreement -------------------------------------------------------------


def test_identical_raters_agree_perfectly():
    labels = ["correct", "incorrect", "correct", "correct"]

    assert cohen_kappa(labels, labels) == pytest.approx(1.0)


def test_a_rater_that_always_says_one_thing_scores_zero():
    """Chance agreement has to be corrected for, or every judge looks good."""
    human = ["correct", "correct", "incorrect", "incorrect"]
    lazy = ["correct", "correct", "correct", "correct"]

    assert cohen_kappa(human, lazy) == pytest.approx(0.0)


def test_a_known_kappa():
    # Two raters, two categories, equal marginals. They agree on four of
    # six items, so the observed agreement is 0.6667. Chance agreement is
    # 0.5, and (0.6667 - 0.5) / (1 - 0.5) is a third.
    first = ["yes", "no", "yes", "no", "yes", "no"]
    second = ["yes", "no", "yes", "no", "no", "yes"]

    assert cohen_kappa(first, second) == pytest.approx(1 / 3, abs=1e-6)


def test_a_single_category_on_both_sides_is_degenerate():
    """Undefined there, so the report says so rather than dividing by zero."""
    assert cohen_kappa(["correct"] * 4, ["correct"] * 4) == 1.0
    assert cohen_kappa(["correct"] * 4, ["incorrect"] * 4) == 0.0


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="same number of items"):
        cohen_kappa(["correct"], ["correct", "incorrect"])


def test_an_empty_set_is_rejected():
    with pytest.raises(ValueError, match="at least one item"):
        cohen_kappa([], [])


def test_the_weighted_kappa_charges_less_for_a_near_miss():
    """A point apart is not the same disagreement as four points apart."""
    identical = [1, 2, 3, 4, 5]
    near = [1, 2, 3, 4, 5]
    far = [5, 4, 3, 2, 1]

    assert weighted_kappa(identical, near) == pytest.approx(1.0)
    assert weighted_kappa(identical, far) < weighted_kappa(identical, near)


def test_an_unknown_weighting_is_rejected():
    with pytest.raises(ValueError, match="Unknown weighting"):
        weighted_kappa([1, 2], [1, 2], weights="cubic")


def test_the_agreement_report_carries_the_confusion_matrix():
    report = agreement(
        ["correct", "correct", "incorrect"],
        ["correct", "incorrect", "incorrect"],
        categories=["incorrect", "correct"],
    )

    assert report["n"] == 3
    assert report["observed_agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["confusion"]["correct"]["correct"] == 1
    assert report["confusion"]["correct"]["incorrect"] == 1
    assert report["interpretation"]


def test_the_kappa_bands():
    assert interpret_kappa(0.9) == "almost perfect"
    assert interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.05) == "slight"
    assert interpret_kappa(-0.2) == "poor"


# --- the transports --------------------------------------------------------


def test_the_ollama_judge_posts_to_the_native_chat_endpoint(monkeypatch):
    seen = capture_ollama(monkeypatch, verdict())

    OllamaJudge("test-model").complete([{"role": "user", "content": "hi"}])

    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["body"]["format"] == "json"
    assert seen["body"]["options"]["seed"] == 42


def test_the_ollama_judge_only_sends_think_when_it_is_set(monkeypatch):
    seen = capture_ollama(monkeypatch, verdict())

    OllamaJudge("m").complete([{"role": "user", "content": "hi"}])
    assert "think" not in seen["body"]

    OllamaJudge("m", think=False).complete([{"role": "user", "content": "hi"}])
    assert seen["body"]["think"] is False


def test_the_openai_compatible_judge_posts_to_the_chat_completions_endpoint(monkeypatch):
    seen = capture_openai(monkeypatch, verdict())

    OpenAICompatJudge("gemini-flash", base_url="https://example.test/v1").complete(
        [{"role": "user", "content": "hi"}]
    )

    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_the_api_key_is_sent_as_a_bearer_token(monkeypatch):
    seen = capture_openai(monkeypatch, verdict())

    OpenAICompatJudge("m", base_url="https://example.test", api_key="secret").complete(
        [{"role": "user", "content": "hi"}]
    )

    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_an_http_error_from_the_judge_is_reported(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            429,
            "Too Many Requests",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(JudgeError, match="429"):
        OpenAICompatJudge("m", base_url="https://example.test").complete(
            [{"role": "user", "content": "hi"}]
        )


def test_an_unreachable_judge_says_so(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(JudgeError, match="ollama serve"):
        OllamaJudge("m").complete([{"role": "user", "content": "hi"}])


def test_the_cache_key_covers_every_setting_that_changes_the_verdict():
    key = OllamaJudge("m", temperature=0.7, seed=7, max_tokens=64, think=False).cache_key

    assert key == {
        "temperature": 0.7,
        "seed": 7,
        "max_tokens": 64,
        "think": False,
        "json_mode": True,
    }


# --- the judge engine ------------------------------------------------------


def test_it_asks_once_per_sample_per_order():
    client = RecordingClient([verdict()] * 6)
    judge = RubricJudge(client, n_samples=3, swap_order=True)

    judge.judge(JudgeItem("Where?", "Bengaluru", "context", "Bengaluru."))

    assert len(client.calls) == 6
    orders = [call[1]["content"].index("ANSWER TO GRADE") for call in client.calls]
    assert len(set(orders)) == 2


def test_one_order_only_when_the_swap_is_off():
    client = RecordingClient([verdict()])
    judge = RubricJudge(client, n_samples=1, swap_order=False)

    judge.judge(JudgeItem("Where?", "Bengaluru", "context", "Bengaluru."))

    assert len(client.calls) == 1


def test_a_failed_sample_is_tolerated_and_counted():
    client = RecordingClient([verdict(), "not a reply", verdict(), verdict()])
    judge = RubricJudge(client, n_samples=2, swap_order=True)

    judgement = judge.judge(JudgeItem("Where?", "Bengaluru", "context", "Bengaluru."))

    assert judgement.n_failed == 1
    assert judgement.n_samples == 3
    assert judgement.correct is True


def test_an_item_the_judge_cannot_read_at_all_is_an_error():
    client = RecordingClient(["not a reply", "still not a reply"])
    judge = RubricJudge(client, n_samples=1, swap_order=True)

    with pytest.raises(JudgeError, match="no readable verdict"):
        judge.judge(JudgeItem("Where?", "Bengaluru", "context", "Bengaluru."))


def test_a_judge_that_needs_at_least_one_sample():
    with pytest.raises(ValueError, match="at least one sample"):
        RubricJudge(RecordingClient([]), n_samples=0)


def test_a_judge_that_changes_its_mind_when_the_order_swaps_is_reported():
    """The same item, graded in both orders, with a verdict that flips."""
    client = RecordingClient([verdict(), verdict(correctness=1, relevance=1)])
    judge = RubricJudge(client, n_samples=1, swap_order=True)

    judgement = judge.judge(JudgeItem("Where?", "Bengaluru", "context", "Bengaluru."))

    assert judgement.order_consistent is False


# --- the cache -------------------------------------------------------------


def test_the_cached_client_serves_a_repeat_from_disk(tmp_path):
    from rageval.cache import DiskCache

    calls: list[str] = []

    class CountingClient:
        name = "counting"
        cache_key: dict[str, Any] = {}

        def complete(self, messages: list[dict[str, str]]) -> str:
            calls.append("called")
            return "a reply"

    client = CachedChatClient(CountingClient(), DiskCache(tmp_path / "cache"))

    client.complete([{"role": "user", "content": "hi"}])
    client.complete([{"role": "user", "content": "hi"}])

    assert calls == ["called"]


# --- the hand labels -------------------------------------------------------


def test_the_shipped_hand_labels_load():
    items = load_labels(LABELS)

    assert len(items) > 20
    assert {item.label for item in items} == {"correct", "incorrect"}
    assert all(item.question for item in items)
    assert all(item.answer for item in items)
    # Three of the items are questions the retriever found nothing for, so
    # the judge is given an empty context. That is the case they describe.
    assert any(not item.context for item in items)


def test_a_held_out_item_is_not_marked_for_tuning():
    items = load_labels(LABELS)

    assert any(item.used_for_tuning for item in items)
    assert any(not item.used_for_tuning for item in items)


def test_a_label_that_is_not_correct_or_incorrect_is_rejected(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "test_id": 1,
                        "question": "q",
                        "reference_answer": "r",
                        "context": "c",
                        "answer": "a",
                        "label": "maybe",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Use one of"):
        load_labels(path)


def test_the_same_answer_for_the_same_case_twice_is_rejected(tmp_path):
    path = tmp_path / "labels.json"
    item = {
        "test_id": 1,
        "question": "q",
        "reference_answer": "r",
        "context": "c",
        "answer": "a",
        "label": "correct",
    }
    path.write_text(json.dumps({"items": [item, item]}), encoding="utf-8")

    with pytest.raises(ValueError, match="repeats the answer"):
        load_labels(path)


def test_labels_round_trip(tmp_path):
    original = LabelledItem(
        test_id=1,
        question="q",
        reference_answer="r",
        context="c",
        answer="a",
        label="correct",
        note="n",
        source="s",
        used_for_tuning=True,
    )
    path = write_labels(tmp_path / "labels.json", [original])

    assert load_labels(path) == [original]


# --- the token rule, which is the thing the judge has to beat ---------------


def test_the_token_rule_misses_the_short_but_right_answers():
    """The cases the token rule gets wrong, and the judge has to get right."""
    reference = "The company uses Python as its primary programming language."
    assert token_rule_verdict("Python", reference) is False
    assert token_rule_verdict("Bengaluru", "Bengaluru, Karnataka.") is False
    assert token_rule_verdict(
        "The standard work week is Monday through Friday.",
        "The standard work week runs Monday through Friday.",
    ) is False


def test_the_token_rule_catches_the_exact_match():
    assert token_rule_verdict("Bengaluru.", "Bengaluru.") is True


def test_the_token_rule_catches_a_matching_number():
    assert token_rule_verdict("18", "18 days") is True


def test_the_token_rule_has_no_verdict_without_a_reference():
    assert token_rule_verdict("I don't know.", "") is False


# --- validation ------------------------------------------------------------


def test_the_validation_compares_the_judge_and_the_token_rule():
    class FixedJudge(RubricJudge):
        def __init__(self, verdicts: dict[str, bool]) -> None:
            self.verdicts = verdicts
            self.name = "fixed"
            self.n_samples = 1
            self.swap_order = False
            self.threshold = 4

        def judge(self, item: JudgeItem) -> Any:
            from rageval.judge.base import Judgement

            correct = self.verdicts.get(item.answer, True)
            score = 5 if correct else 1
            return Judgement(
                correct=correct,
                correctness=score,
                relevance=score,
                groundedness=score,
                completeness=score,
            )

    items = [
        LabelledItem(
            test_id=1,
            question="Which city?",
            reference_answer="Bengaluru, Karnataka.",
            context="The office is in Bengaluru.",
            answer="Bengaluru",
            label="correct",
        ),
        LabelledItem(
            test_id=2,
            question="How many people?",
            reference_answer="Approximately 120 people.",
            context="The company employs approximately 120 people.",
            answer="18",
            label="incorrect",
        ),
    ]

    report = validate(FixedJudge({"Bengaluru": True, "18": False}), items)

    # The judge gets both right; the token rule misses the first one.
    assert report["all"]["judge"]["cohen_kappa"] == pytest.approx(1.0)
    assert report["all"]["token_rule"]["observed_agreement"] == pytest.approx(0.5)


def test_the_validation_reports_the_held_out_subset():
    class AlwaysCorrect(RubricJudge):
        def __init__(self) -> None:
            self.name = "always"
            self.n_samples = 1
            self.swap_order = False
            self.threshold = 4

        def judge(self, item: JudgeItem) -> Any:
            from rageval.judge.base import Judgement

            return Judgement(
                correct=True,
                correctness=5,
                relevance=5,
                groundedness=5,
                completeness=5,
            )

    items = [
        LabelledItem(
            test_id=1,
            question="q",
            reference_answer="r",
            context="c",
            answer="a",
            label="correct",
            used_for_tuning=True,
        ),
        LabelledItem(
            test_id=2,
            question="q",
            reference_answer="r",
            context="c",
            answer="b",
            label="correct",
        ),
    ]

    report = validate(AlwaysCorrect(), items)

    assert report["all"]["n_items"] == 2
    assert report["held_out"]["n_items"] == 1


# --- wiring into the runner ------------------------------------------------


def test_the_runner_builds_a_local_judge(config: EvalConfig):
    config.use_cache = False
    config.judge.provider = "ollama"
    config.judge.model = "llama3.2:3b"

    judge = build_judge(config)

    assert judge.name == "ollama:llama3.2:3b"
    assert judge.n_samples == config.judge.n_samples
    assert judge.swap_order is config.judge.swap_order


def test_the_local_judge_does_not_need_a_key(config: EvalConfig):
    config.use_cache = False
    config.judge.provider = "ollama"
    config.judge.model = "llama3.2:3b"

    assert build_judge(config) is not None


def test_the_hosted_judge_needs_a_key(config: EvalConfig, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config.use_cache = False
    config.judge.provider = "gemini"

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_judge(config)


def test_the_hosted_judge_reads_its_key_from_the_environment(
    config: EvalConfig,
    monkeypatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", "s3cret")
    config.use_cache = False
    config.judge.provider = "gemini"
    config.judge.model = "gemini-3.8-flash"

    judge = build_judge(config)

    assert judge.name == "openai-compat:gemini-3.8-flash"


def test_an_unknown_judge_provider_is_rejected(config: EvalConfig):
    config.use_cache = False
    config.judge.provider = "mystery"

    with pytest.raises(ValueError, match="Unknown judge provider"):
        build_judge(config)


def test_the_judge_adds_its_scores_to_the_case(
    config: EvalConfig,
    knowledge_base,
    dataset_path,
):
    config.dataset_path = dataset_path
    config.knowledge_base_path = knowledge_base
    config.use_cache = False

    class FixedJudge(RubricJudge):
        def __init__(self) -> None:
            self.name = "fixed"
            self.n_samples = 1
            self.swap_order = False
            self.threshold = 4

        def judge(self, item: JudgeItem) -> Any:
            from rageval.judge.base import Judgement

            return Judgement(
                correct=True,
                correctness=5,
                relevance=4,
                groundedness=5,
                completeness=3,
            )

    from rageval.generation.extractive import ExtractiveGenerator
    from rageval.retrieval.lexical import (
        BM25Retriever,
        build_chunks,
        split_into_sentences,
    )

    case = load_dataset(config.dataset_path).cases[0]
    document = knowledge_base.read_text(encoding="utf-8")
    chunks = build_chunks(split_into_sentences(document))
    result = evaluate_case(
        case,
        BM25Retriever(chunks),
        ExtractiveGenerator(),
        config,
        FixedJudge(),
    )

    assert result.metrics["judge_correct"] == 1.0
    assert result.metrics["judge_correctness"] == 5.0
    assert result.metrics["judge_relevance"] == 4.0
    assert result.metrics["judge_completeness"] == 3.0
    assert result.judge["correct"] is True


def test_the_judge_and_the_token_rule_are_compared_per_case(
    config: EvalConfig,
    knowledge_base,
    dataset_path,
):
    config.dataset_path = dataset_path
    config.knowledge_base_path = knowledge_base
    config.use_cache = False

    class WrongJudge(RubricJudge):
        def __init__(self) -> None:
            self.name = "fixed"
            self.n_samples = 1
            self.swap_order = False
            self.threshold = 4

        def judge(self, item: JudgeItem) -> Any:
            from rageval.judge.base import Judgement

            return Judgement(
                correct=False,
                correctness=1,
                relevance=1,
                groundedness=1,
                completeness=1,
            )

    from rageval.generation.extractive import ExtractiveGenerator
    from rageval.retrieval.lexical import (
        BM25Retriever,
        build_chunks,
        split_into_sentences,
    )

    case = load_dataset(config.dataset_path).cases[0]
    document = knowledge_base.read_text(encoding="utf-8")
    chunks = build_chunks(split_into_sentences(document))
    result = evaluate_case(
        case,
        BM25Retriever(chunks),
        ExtractiveGenerator(),
        config,
        WrongJudge(),
    )

    assert result.metrics["judge_token_agreement"] == 0.0
