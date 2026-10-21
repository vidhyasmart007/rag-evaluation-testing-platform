"""Tests for the embedding clients.

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
from rageval.retrieval.embeddings import (
    CachedEmbedder,
    EmbeddingError,
    OllamaEmbedder,
)


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


def embedding_payload(count: int, dims: int = 3) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": [0.1] * dims}
            for index in range(count)
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


class FakeEmbedder:
    """An embedder that maps text to a fixed vector, without any network."""

    name = "fake"

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[list[str]] = []

    def embed(self, texts):
        items = list(texts)
        self.calls.append(items)
        return [
            self.vectors.get(text, [float(len(text) % 7), 1.0, 0.5]) for text in items
        ]


def test_it_posts_to_the_openai_compatible_endpoint(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["headers"] = request.headers
        seen["timeout"] = timeout
        return FakeResponse(embedding_payload(1))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedder = OllamaEmbedder("test-model", base_url="http://localhost:11434/v1")
    vectors = embedder.embed(["hello"])

    assert seen["url"] == "http://localhost:11434/v1/embeddings"
    assert seen["body"] == {"model": "test-model", "input": ["hello"]}
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["headers"]["Authorization"] == "Bearer ollama"
    assert vectors == [[0.1, 0.1, 0.1]]


def test_a_configured_api_key_is_sent(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_urlopen(request, timeout=None):
        seen["headers"] = request.headers
        return FakeResponse(embedding_payload(1))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    OllamaEmbedder("m", api_key="secret").embed(["hi"])

    assert seen["headers"]["Authorization"] == "Bearer secret"


def test_it_returns_one_vector_per_input(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: FakeResponse(embedding_payload(3)),
    )

    vectors = OllamaEmbedder("m").embed(["a", "b", "c"])

    assert len(vectors) == 3


def test_it_orders_the_vectors_by_the_index_field(monkeypatch):
    payload = {
        "data": [
            {"index": 2, "embedding": [3.0]},
            {"index": 0, "embedding": [1.0]},
            {"index": 1, "embedding": [2.0]},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: FakeResponse(payload),
    )

    vectors = OllamaEmbedder("m").embed(["a", "b", "c"])

    assert vectors == [[1.0], [2.0], [3.0]]


def test_it_splits_large_inputs_into_batches(monkeypatch):
    batches: list[list[str]] = []

    def fake_urlopen(request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        batches.append(body["input"])
        return FakeResponse(embedding_payload(len(body["input"])))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    OllamaEmbedder("m", batch_size=2).embed(["a", "b", "c", "d", "e"])

    assert batches == [["a", "b"], ["c", "d"], ["e"]]


def test_an_empty_input_makes_no_request(monkeypatch):
    def explode(request, timeout=None):  # pragma: no cover - must not run
        raise AssertionError("no request should be made")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    assert OllamaEmbedder("m").embed([]) == []


def test_the_name_carries_the_model_so_caches_cannot_collide():
    assert OllamaEmbedder("model-a").name == "ollama:model-a"
    assert OllamaEmbedder("model-b").name == "ollama:model-b"
    assert OllamaEmbedder("model-a").name != OllamaEmbedder("model-b").name


def test_an_http_error_is_reported_with_the_detail(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            "http://localhost:11434/v1/embeddings",
            404,
            "Not Found",
            {},
            None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(EmbeddingError, match="404"):
        OllamaEmbedder("missing-model").embed(["hello"])


def test_a_connection_error_explains_how_to_start_ollama(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(EmbeddingError, match="ollama serve"):
        OllamaEmbedder("m").embed(["hello"])


def test_a_timeout_is_reported(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise TimeoutError()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(EmbeddingError, match="did not respond"):
        OllamaEmbedder("m").embed(["hello"])


def test_a_short_response_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: FakeResponse(embedding_payload(1)),
    )

    with pytest.raises(EmbeddingError, match="2 input"):
        OllamaEmbedder("m").embed(["a", "b"])


def test_an_empty_embedding_is_rejected(monkeypatch):
    payload = {"data": [{"index": 0, "embedding": []}]}
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout=None: FakeResponse(payload),
    )

    with pytest.raises(EmbeddingError, match="empty embedding"):
        OllamaEmbedder("m").embed(["a"])


def test_the_cached_embedder_reuses_a_stored_vector(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    inner = FakeEmbedder({"hello": [1.0, 0.0, 0.0]})
    embedder = CachedEmbedder(inner, cache)

    first = embedder.embed(["hello"])
    second = embedder.embed(["hello"])

    assert first == second == [[1.0, 0.0, 0.0]]
    assert len(inner.calls) == 1


def test_the_cached_embedder_computes_only_what_is_missing(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    inner = FakeEmbedder()
    embedder = CachedEmbedder(inner, cache)

    embedder.embed(["alpha"])
    embedder.embed(["alpha", "beta"])

    assert inner.calls == [["alpha"], ["beta"]]


def test_the_cached_embedder_preserves_the_order_of_the_input(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    inner = FakeEmbedder(
        {"a": [1.0], "b": [2.0], "c": [3.0]},
    )
    embedder = CachedEmbedder(inner, cache)

    vectors = embedder.embed(["c", "a", "b"])

    assert vectors == [[3.0], [1.0], [2.0]]


def test_two_models_do_not_share_a_cached_vector(tmp_path):
    cache = DiskCache(tmp_path / "cache")

    first = FakeEmbedder({"hello": [1.0]})
    second = FakeEmbedder({"hello": [2.0]})
    first.name = "fake:a"
    second.name = "fake:b"

    CachedEmbedder(first, cache).embed(["hello"])
    vectors = CachedEmbedder(second, cache).embed(["hello"])

    assert vectors == [[2.0]]


def test_a_changed_text_is_a_different_entry(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    inner = FakeEmbedder({"hello": [1.0], "other": [2.0]})
    embedder = CachedEmbedder(inner, cache)

    embedder.embed(["hello"])
    vectors = embedder.embed(["other"])

    assert vectors == [[2.0]]


def test_the_cached_embedder_keeps_the_wrapped_name(tmp_path):
    embedder = CachedEmbedder(FakeEmbedder(), DiskCache(tmp_path / "cache"))
    assert embedder.name == "fake"


def test_the_cached_embedder_of_an_empty_input_is_empty(tmp_path):
    embedder = CachedEmbedder(FakeEmbedder(), DiskCache(tmp_path / "cache"))
    assert embedder.embed([]) == []


def test_a_wrong_number_of_vectors_is_reported(tmp_path):
    class ShortEmbedder:
        name = "short"

        def embed(self, texts):
            return [[1.0]]

    embedder = CachedEmbedder(ShortEmbedder(), DiskCache(tmp_path / "cache"))

    with pytest.raises(EmbeddingError, match="Expected 2 embeddings"):
        embedder.embed(["a", "b"])
