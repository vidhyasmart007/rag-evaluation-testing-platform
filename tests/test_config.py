"""Tests for configuration handling."""

from __future__ import annotations

import json

import pytest

from rageval.config import EvalConfig, GenerationConfig, RetrievalConfig


def test_the_defaults_are_usable():
    config = EvalConfig()
    assert config.retrieval.retriever == "lexical"
    assert config.generation.provider == "extractive"
    assert config.generation.temperature == 0.0
    assert config.judge.enabled is False


def test_from_dict_builds_the_nested_sections():
    config = EvalConfig.from_dict(
        {
            "retrieval": {"top_k": 5, "min_score": 0.4},
            "generation": {"provider": "extractive", "min_answer_score": 0.6},
        }
    )
    assert config.retrieval.top_k == 5
    assert config.retrieval.min_score == 0.4
    assert config.generation.min_answer_score == 0.6


def test_an_unknown_option_is_rejected():
    with pytest.raises(ValueError, match="Unknown option"):
        EvalConfig.from_dict({"retrieval": {"top_kk": 5}})


def test_an_unknown_section_option_is_rejected():
    with pytest.raises(ValueError, match="Unknown option"):
        EvalConfig.from_dict({"generation": {"model_name": "x"}})


def test_an_unknown_top_level_option_is_rejected():
    with pytest.raises(ValueError, match="Unknown config option"):
        EvalConfig.from_dict({"retrieval": {}, "nonsense": 1})


def test_overrides_apply_dotted_paths():
    config = EvalConfig()
    config.apply_overrides(
        [
            "retrieval.top_k=5",
            "retrieval.min_score=0.35",
            "generation.provider=extractive",
            "use_cache=false",
        ]
    )
    assert config.retrieval.top_k == 5
    assert config.retrieval.min_score == pytest.approx(0.35)
    assert config.generation.provider == "extractive"
    assert config.use_cache is False


def test_an_override_without_an_equals_sign_is_rejected():
    with pytest.raises(ValueError, match="Expected the form"):
        EvalConfig().apply_overrides(["retrieval.top_k"])


def test_an_override_for_an_unknown_section_is_rejected():
    with pytest.raises(ValueError, match="Unknown config section"):
        EvalConfig().apply_overrides(["nonsense.value=1"])


def test_an_override_for_an_unknown_option_is_rejected():
    with pytest.raises(ValueError, match="Unknown config option"):
        EvalConfig().apply_overrides(["retrieval.nonsense=1"])


def test_a_deeply_nested_override_is_rejected():
    with pytest.raises(ValueError, match="Invalid override path"):
        EvalConfig().apply_overrides(["a.b.c=1"])


def test_to_dict_round_trips():
    original = EvalConfig()
    original.retrieval.top_k = 7
    restored = EvalConfig.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.retrieval.top_k == 7


def test_load_reads_a_json_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"retrieval": {"top_k": 9}}), encoding="utf-8")
    assert EvalConfig.load(path).retrieval.top_k == 9


def test_load_of_a_missing_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        EvalConfig.load(tmp_path / "absent.json")


def test_load_without_a_path_returns_the_defaults():
    assert EvalConfig.load().retrieval.top_k == RetrievalConfig().top_k


def test_api_keys_are_referenced_by_name_and_never_stored():
    """A configuration file gets committed, so it must not hold secrets."""
    config = EvalConfig()
    config.generation.api_key_env = "OLLAMA_API_KEY"
    config.judge.api_key_env = "GEMINI_API_KEY"

    serialised = json.dumps(config.to_dict())

    assert "OLLAMA_API_KEY" in serialised
    assert "GEMINI_API_KEY" in serialised
    assert "api_key" not in serialised.replace("api_key_env", "")
    assert "api_key\"" not in serialised


def test_the_generation_config_has_no_plain_api_key_field():
    assert "api_key" not in {field for field in GenerationConfig().to_dict()}


def test_the_dense_section_has_usable_defaults():
    config = EvalConfig()
    assert config.dense.model == "embeddinggemma"
    assert config.dense.base_url.startswith("http")
    assert config.dense.api_key_env is None


def test_the_dense_section_can_be_overridden():
    config = EvalConfig()
    config.apply_overrides(
        [
            "dense.model=bge-m3",
            "dense.base_url=http://localhost:11434/v1",
            "dense.batch_size=4",
        ]
    )
    assert config.dense.model == "bge-m3"
    assert config.dense.batch_size == 4


def test_each_retriever_has_its_own_relevance_floor():
    lexical = EvalConfig()
    lexical.retrieval.retriever = "lexical"
    dense = EvalConfig()
    dense.retrieval.retriever = "dense"
    hybrid = EvalConfig()
    hybrid.retrieval.retriever = "hybrid"

    assert lexical.retrieval.resolved_min_score == 0.30
    assert dense.retrieval.resolved_min_score == 0.35
    assert hybrid.retrieval.resolved_min_score == 0.40


def test_the_floor_can_be_overridden_for_the_active_retriever():
    config = EvalConfig()
    config.apply_overrides(["retrieval.retriever=dense", "retrieval.min_score=0.6"])

    assert config.retrieval.resolved_min_score == 0.6


def test_the_report_records_the_floor_that_was_applied():
    config = EvalConfig()
    config.retrieval.retriever = "hybrid"

    assert config.to_dict()["retrieval"]["min_score"] == 0.40


def test_a_read_only_property_cannot_be_overridden():
    config = EvalConfig()
    with pytest.raises(ValueError, match="Unknown config option"):
        config.apply_overrides(["retrieval.resolved_min_score=0.9"])


def test_the_hybrid_weights_are_configurable():
    config = EvalConfig()
    config.apply_overrides(
        ["retrieval.lexical_weight=0.7", "retrieval.dense_weight=0.3"]
    )

    assert config.retrieval.lexical_weight == 0.7
    assert config.retrieval.dense_weight == 0.3
