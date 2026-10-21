"""Tests for the response cache and the cached generator wrapper."""

from __future__ import annotations

import pytest

from rageval.cache import DiskCache, framework_fingerprint
from rageval.generation.base import GenerationResult
from rageval.generation.cached import CachedGenerator
from rageval.generation.extractive import ExtractiveGenerator

CONTEXT = (
    "The company office is located in Bengaluru, Karnataka. "
    "Employees receive 18 annual leave days per calendar year."
)


def test_a_miss_computes_and_stores(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    calls: list[str] = []

    def compute() -> dict[str, object]:
        calls.append("called")
        return {"text": "Bengaluru"}

    value, was_cached = cache.get_or_compute("key", compute)

    assert value == {"text": "Bengaluru"}
    assert was_cached is False
    assert calls == ["called"]


def test_a_hit_does_not_recompute(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    calls: list[str] = []

    def compute() -> dict[str, object]:
        calls.append("called")
        return {"text": "Bengaluru"}

    cache.get_or_compute("key", compute)
    value, was_cached = cache.get_or_compute("key", compute)

    assert value == {"text": "Bengaluru"}
    assert was_cached is True
    assert calls == ["called"]


def test_a_disabled_cache_never_stores(tmp_path):
    cache = DiskCache(tmp_path / "cache", enabled=False)

    assert cache.get("key") is None
    cache.set("key", {"text": "Bengaluru"})
    assert cache.get("key") is None
    assert not (tmp_path / "cache").exists()


def test_a_corrupt_entry_is_treated_as_a_miss(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    cache.set("key", {"text": "Bengaluru"})
    (tmp_path / "cache" / "key.json").write_text("{not json", encoding="utf-8")

    assert cache.get("key") is None


def test_the_key_depends_on_every_part(tmp_path):
    first = DiskCache.make_key("ns", model="a", question="q", context="c")
    same = DiskCache.make_key("ns", model="a", question="q", context="c")
    other = DiskCache.make_key("ns", model="a", question="q", context="different")

    assert first == same
    assert first != other


def test_the_framework_fingerprint_is_stable():
    assert framework_fingerprint() == framework_fingerprint()
    assert len(framework_fingerprint()) == 16


def test_the_cached_generator_serves_the_second_call_from_disk(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    generator = CachedGenerator(ExtractiveGenerator(), cache)

    first = generator.generate("Where is the office?", CONTEXT)
    second = generator.generate("Where is the office?", CONTEXT)

    assert first.cached is False
    assert second.cached is True
    assert second.text == first.text
    assert second.abstained == first.abstained


def test_the_cached_generator_does_not_serve_a_different_context(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    generator = CachedGenerator(ExtractiveGenerator(), cache)

    generator.generate("Where is the office?", CONTEXT)
    other = generator.generate("Where is the office?", "Completely different text.")

    assert other.cached is False


def test_the_cached_generator_preserves_an_abstention(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    generator = CachedGenerator(ExtractiveGenerator(), cache)

    first = generator.generate("Who is the chief executive officer?", CONTEXT)
    second = generator.generate("Who is the chief executive officer?", CONTEXT)

    assert first.abstained is True
    assert second.abstained is True
    assert second.cached is True


def test_the_cached_generator_keeps_the_wrapped_generator_name(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    generator = CachedGenerator(ExtractiveGenerator(), cache)

    assert generator.name == "extractive"


def test_the_cache_is_not_used_when_it_is_disabled(tmp_path):
    from rageval.config import EvalConfig
    from rageval.runner import build_generator

    config = EvalConfig(use_cache=False, cache_dir=tmp_path / "cache")
    generator = build_generator(config)

    result = generator.generate("Where is the office?", CONTEXT)

    assert result.cached is False
    assert not (tmp_path / "cache").exists()


def test_a_run_reports_its_cache_hits(config):
    from rageval.runner import run_evaluation

    first = run_evaluation(config)
    second = run_evaluation(config)

    assert first.aggregates["cache_hits"] == 0.0
    assert second.aggregates["cache_hits"] == float(len(second.cases))
    assert all(case.cached for case in second.cases)


def test_a_cached_run_produces_the_same_scores(config):
    from rageval.runner import run_evaluation

    first = run_evaluation(config)
    second = run_evaluation(config)

    assert first.aggregates["answer_correct"] == second.aggregates["answer_correct"]
    assert first.aggregates["hit_at_k"] == second.aggregates["hit_at_k"]
    assert [case.answer for case in first.cases] == [
        case.answer for case in second.cases
    ]


def test_a_generation_result_round_trips_through_the_cache(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    key = DiskCache.make_key("ns", model="m", question="q", context="c")
    original = GenerationResult(
        text="Bengaluru",
        model="m",
        prompt_tokens=11,
        completion_tokens=7,
        abstained=False,
    )
    cache.set(
        key,
        {
            "text": original.text,
            "model": original.model,
            "prompt_tokens": original.prompt_tokens,
            "completion_tokens": original.completion_tokens,
            "abstained": original.abstained,
        },
    )

    restored = cache.get(key)

    assert restored is not None
    assert restored["text"] == original.text
    assert restored["prompt_tokens"] == original.prompt_tokens
    assert restored["completion_tokens"] == original.completion_tokens


def test_the_cache_directory_is_created_lazily(tmp_path):
    DiskCache(tmp_path / "nested" / "cache")
    assert (tmp_path / "nested" / "cache").is_dir()


def test_an_unknown_provider_is_still_rejected_before_the_cache_is_built(config):
    from rageval.runner import build_generator

    config.generation.provider = "magic"
    with pytest.raises(ValueError, match="Unknown generation provider"):
        build_generator(config)
