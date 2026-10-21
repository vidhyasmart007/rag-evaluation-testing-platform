"""Tests for the language model generator.

None of these tests need a running model server. The HTTP layer is
replaced, so the suite still passes on a machine where Ollama is not
installed.
"""

from __future__ import annotations

import json
import urllib.error
from typing import Any

import pytest

from rageval.cache import DiskCache
from rageval.config import EvalConfig
from rageval.generation.base import ABSTAINED_ANSWER, GenerationError
from rageval.generation.cached import CachedGenerator
from rageval.generation.extractive import ExtractiveGenerator
from rageval.generation.ollama import (
    OllamaGenerator,
    extract_json_object,
    looks_like_refusal,
)
from rageval.runner import build_generator


class FakeResponse:
    """Stands in for the object returned by urlopen."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def completion(content: str, **usage: int) -> dict[str, Any]:
    """Build an Ollama chat response."""
    body: dict[str, Any] = {
        "model": "test-model",
        "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    if usage:
        body["prompt_eval_count"] = usage.get("prompt_tokens", 0)
        body["eval_count"] = usage.get("completion_tokens", 0)
    return body


def answer_json(answer: str, abstained: bool = False) -> str:
    return json.dumps({"answer": answer, "abstained": abstained})


def capture(monkeypatch, content: str, **usage: int) -> dict[str, Any]:
    """Capture the request a generator makes and answer it with ``content``."""
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["headers"] = request.headers
        seen["timeout"] = timeout
        return FakeResponse(completion(content, **usage))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


# --- the request -----------------------------------------------------------


def test_it_posts_to_the_native_chat_endpoint(monkeypatch):
    seen = capture(monkeypatch, answer_json("Bengaluru."))

    OllamaGenerator("test-model").generate("Where?", "The office is in Bengaluru.")

    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["headers"]["Authorization"] == "Bearer ollama"


def test_the_request_carries_the_reproducibility_settings(monkeypatch):
    seen = capture(monkeypatch, answer_json("Bengaluru."))

    OllamaGenerator("m", temperature=0.0, seed=42, max_tokens=128).generate(
        "Where?", "text"
    )

    assert seen["body"]["model"] == "m"
    assert seen["body"]["stream"] is False
    assert seen["body"]["format"] == "json"
    assert seen["body"]["options"] == {
        "temperature": 0.0,
        "seed": 42,
        "num_predict": 128,
    }


def test_the_think_parameter_is_only_sent_when_it_is_set(monkeypatch):
    seen = capture(monkeypatch, answer_json("Bengaluru."))

    OllamaGenerator("m").generate("Where?", "text")
    assert "think" not in seen["body"]

    OllamaGenerator("m", think=False).generate("Where?", "text")
    assert seen["body"]["think"] is False

def test_the_prompt_numbers_the_sources(monkeypatch):
    seen = capture(monkeypatch, answer_json("Bengaluru."))

    OllamaGenerator("m").generate(
        "Where?",
        "The office is in Bengaluru.\nEmployees receive 18 days of leave.",
    )

    messages = seen["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert "abstained" in messages[0]["content"]
    assert "[1] The office is in Bengaluru." in messages[1]["content"]
    assert "[2] Employees receive 18 days of leave." in messages[1]["content"]
    assert "Question: Where?" in messages[1]["content"]


def test_a_configured_api_key_is_sent(monkeypatch):
    seen = capture(monkeypatch, answer_json("Bengaluru."))

    OllamaGenerator("m", api_key="secret").generate("Where?", "text")

    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_the_timeout_is_passed_to_the_http_layer(monkeypatch):
    seen = capture(monkeypatch, answer_json("Bengaluru."))

    OllamaGenerator("m", timeout=12.5).generate("Where?", "text")

    assert seen["timeout"] == 12.5


# --- reading the answer ----------------------------------------------------


def test_it_reads_an_answer_out_of_a_json_response(monkeypatch):
    capture(monkeypatch, answer_json("The office is in Bengaluru."))

    result = OllamaGenerator("m").generate("Where?", "The office is in Bengaluru.")

    assert result.text == "The office is in Bengaluru."
    assert result.abstained is False
    assert result.reason == ""


def test_it_records_a_refusal(monkeypatch):
    capture(monkeypatch, answer_json("", abstained=True))

    result = OllamaGenerator("m").generate("Where?", "Some context.")

    assert result.abstained is True
    assert result.text == ABSTAINED_ANSWER
    assert result.reason == "model_refusal"


def test_it_accepts_a_boolean_written_as_a_string(monkeypatch):
    capture(monkeypatch, '{"answer": "", "abstained": "true"}')

    result = OllamaGenerator("m").generate("Where?", "Some context.")

    assert result.abstained is True


def test_it_finds_json_embedded_in_prose(monkeypatch):
    capture(monkeypatch, 'Sure! Here you go: {"answer": "Bengaluru."} Hope that helps.')

    result = OllamaGenerator("m").generate("Where?", "The office is in Bengaluru.")

    assert result.text == "Bengaluru."
    assert result.abstained is False


def test_it_finds_json_inside_a_code_fence(monkeypatch):
    capture(monkeypatch, '```json\n{"answer": "Bengaluru.", "abstained": false}\n```')

    result = OllamaGenerator("m").generate("Where?", "The office is in Bengaluru.")

    assert result.text == "Bengaluru."


def test_an_empty_answer_is_treated_as_a_refusal(monkeypatch):
    capture(monkeypatch, '{"answer": "", "abstained": false}')

    result = OllamaGenerator("m").generate("Where?", "Some context.")

    assert result.abstained is True
    assert result.reason == "empty_answer"


def test_a_refusal_written_as_the_answer_is_still_a_refusal(monkeypatch):
    capture(monkeypatch, '{"answer": "I do not know.", "abstained": false}')

    result = OllamaGenerator("m").generate("Where?", "Some context.")

    assert result.abstained is True
    assert result.reason == "model_refusal"


def test_text_that_is_not_json_is_used_as_the_answer(monkeypatch):
    capture(monkeypatch, "The office is in Bengaluru, Karnataka.")

    result = OllamaGenerator("m").generate("Where?", "The office is in Bengaluru.")

    assert result.text == "The office is in Bengaluru, Karnataka."
    assert result.abstained is False


def test_unparseable_refusal_text_is_treated_as_a_refusal(monkeypatch):
    capture(monkeypatch, "I don't know based on the sources provided.")

    result = OllamaGenerator("m").generate("Where?", "Some context.")

    assert result.abstained is True
    assert result.reason == "model_refusal"


def test_an_empty_response_is_treated_as_a_refusal(monkeypatch):
    capture(monkeypatch, "")

    result = OllamaGenerator("m").generate("Where?", "Some context.")

    assert result.abstained is True
    assert result.reason == "empty_answer"


def test_it_records_the_token_usage(monkeypatch):
    capture(monkeypatch, answer_json("Bengaluru."), prompt_tokens=31, completion_tokens=7)

    result = OllamaGenerator("m").generate("Where?", "The office is in Bengaluru.")

    assert result.prompt_tokens == 31
    assert result.completion_tokens == 7


# --- abstention without a model call ---------------------------------------


def test_an_empty_context_abstains_without_calling_the_model(monkeypatch):
    def explode(request, timeout=None):  # pragma: no cover - must not run
        raise AssertionError("no request should be made")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    result = OllamaGenerator("m").generate("Where?", "   ")

    assert result.abstained is True
    assert result.text == ABSTAINED_ANSWER
    assert result.reason == "empty_context"


# --- failures --------------------------------------------------------------


def test_an_http_error_is_reported_with_the_detail(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "http://localhost:11434/v1/chat/completions",
            404,
            "Not Found",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(GenerationError, match="404"):
        OllamaGenerator("missing-model").generate("Where?", "text")


def test_a_connection_error_explains_how_to_start_ollama(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(GenerationError, match="ollama serve"):
        OllamaGenerator("m").generate("Where?", "text")


def test_a_timeout_is_reported(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(GenerationError, match="did not respond"):
        OllamaGenerator("m").generate("Where?", "text")


def test_a_response_that_is_not_json_is_reported(monkeypatch):
    class NotJson:
        def read(self) -> bytes:
            return b"<html>gateway error</html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout=None: NotJson())

    with pytest.raises(GenerationError, match="not valid JSON"):
        OllamaGenerator("m").generate("Where?", "text")


# --- the cache key ---------------------------------------------------------


def test_the_cache_key_covers_every_setting_that_changes_the_answer():
    generator = OllamaGenerator("m", temperature=0.7, seed=7, max_tokens=64, think=False)

    assert generator.cache_key == {
        "temperature": 0.7,
        "seed": 7,
        "max_tokens": 64,
        "think": False,
    }


def test_two_temperatures_do_not_share_a_cached_answer(tmp_path):
    """A cached answer made at one temperature is not an answer at another."""
    cache = DiskCache(tmp_path / "cache")
    calls: list[str] = []

    class RecordingGenerator:
        name = "ollama:recording"

        def __init__(self, temperature: float) -> None:
            self.temperature = temperature
            self.cache_key = {"temperature": temperature}

        def generate(self, question, context):
            calls.append(self.temperature)
            from rageval.generation.base import GenerationResult

            return GenerationResult(text="Bengaluru.", model=self.name)

    CachedGenerator(RecordingGenerator(0.0), cache).generate("Where?", "text")
    CachedGenerator(RecordingGenerator(0.9), cache).generate("Where?", "text")

    assert calls == [0.0, 0.9]


def test_the_same_settings_do_share_a_cached_answer(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    calls: list[str] = []

    class RecordingGenerator:
        name = "ollama:recording"

        def __init__(self) -> None:
            self.cache_key = {"temperature": 0.0}

        def generate(self, question, context):
            calls.append("called")
            from rageval.generation.base import GenerationResult

            return GenerationResult(text="Bengaluru.", model=self.name)

    CachedGenerator(RecordingGenerator(), cache).generate("Where?", "text")
    second = CachedGenerator(RecordingGenerator(), cache).generate("Where?", "text")

    assert calls == ["called"]
    assert second.cached is True


def test_a_cached_answer_keeps_its_abstention_reason(tmp_path):
    cache = DiskCache(tmp_path / "cache")

    class RefusingGenerator:
        name = "ollama:refusing"

        def __init__(self) -> None:
            self.cache_key: dict[str, Any] = {}

        def generate(self, question, context):
            from rageval.generation.base import GenerationResult

            return GenerationResult(
                text=ABSTAINED_ANSWER,
                model=self.name,
                abstained=True,
                reason="model_refusal",
            )

    CachedGenerator(RefusingGenerator(), cache).generate("Where?", "text")
    second = CachedGenerator(RefusingGenerator(), cache).generate("Where?", "text")

    assert second.abstained is True
    assert second.reason == "model_refusal"


def test_the_extractive_generator_covers_its_threshold_in_the_cache_key():
    assert ExtractiveGenerator(min_answer_score=0.45).cache_key == {
        "min_answer_score": 0.45
    }


def test_two_thresholds_do_not_share_a_cached_answer(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    # "sick" is absent from the context, so the coverage is 0.8: above the
    # low threshold and below the high one.
    question = "How many sick leave days do employees receive?"
    context = "Employees receive 18 annual leave days per calendar year."

    low = ExtractiveGenerator(min_answer_score=0.1)
    high = ExtractiveGenerator(min_answer_score=0.9)

    first = CachedGenerator(low, cache).generate(question, context)
    second = CachedGenerator(high, cache).generate(question, context)

    assert first.abstained is False
    assert second.abstained is True


# --- the helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"answer": "x", "abstained": false}', {"answer": "x", "abstained": False}),
        ('```json\n{"answer": "x"}\n```', {"answer": "x"}),
        ('prefix {"answer": "x"} suffix', {"answer": "x"}),
    ],
)
def test_json_is_extracted_from_the_shapes_models_produce(text, expected):
    assert extract_json_object(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "{not json}"])
def test_text_without_json_yields_nothing(text):
    assert extract_json_object(text) is None


@pytest.mark.parametrize(
    "text",
    ["I don't know.", "There is not enough information.", "Unable to answer."],
)
def test_refusal_wording_is_recognised(text):
    assert looks_like_refusal(text) is True


def test_an_ordinary_answer_is_not_a_refusal():
    assert looks_like_refusal("The office is in Bengaluru, Karnataka.") is False


# --- wiring into the runner ------------------------------------------------


def test_the_runner_builds_an_ollama_generator(config: EvalConfig):
    config.use_cache = False
    config.generation.provider = "ollama"
    config.generation.model = "qwen3:1.7b"

    generator = build_generator(config)

    assert isinstance(generator, OllamaGenerator)
    assert generator.model == "qwen3:1.7b"
    assert generator.name == "ollama:qwen3:1.7b"


def test_the_ollama_provider_needs_a_model(config: EvalConfig):
    config.generation.provider = "ollama"

    with pytest.raises(ValueError, match="generation.model"):
        build_generator(config)


def test_an_unknown_provider_is_rejected(config: EvalConfig):
    config.generation.provider = "mystery"

    with pytest.raises(ValueError, match="Available: extractive, ollama"):
        build_generator(config)


def test_the_configured_base_url_is_used(config: EvalConfig):
    config.use_cache = False
    config.generation.provider = "ollama"
    config.generation.model = "m"
    config.generation.base_url = "http://example.test/v1"

    generator = build_generator(config)

    assert generator.base_url == "http://example.test/v1"


def test_the_api_key_is_read_from_the_environment(config: EvalConfig, monkeypatch):
    monkeypatch.setenv("RAGEVAL_TEST_KEY", "s3cret")
    config.use_cache = False
    config.generation.provider = "ollama"
    config.generation.model = "m"
    config.generation.api_key_env = "RAGEVAL_TEST_KEY"

    generator = build_generator(config)

    assert generator.api_key == "s3cret"


def test_the_cached_generator_keeps_the_wrapped_name(config: EvalConfig):
    config.generation.provider = "ollama"
    config.generation.model = "m"

    generator = build_generator(config)

    assert isinstance(generator, CachedGenerator)
    assert generator.name == "ollama:m"
