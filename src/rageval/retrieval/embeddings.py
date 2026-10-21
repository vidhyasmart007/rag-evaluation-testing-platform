"""Embedding clients for the dense retriever.

Ollama exposes an OpenAI compatible endpoint at ``/v1/embeddings``, so the
client here needs nothing beyond the standard library. The project keeps
zero runtime dependencies on purpose, and one POST with a JSON body is not
worth a dependency.

Two implementations are provided:

* :class:`OllamaEmbedder` calls a running Ollama server.
* :class:`CachedEmbedder` memoises embeddings on disk, so a corpus is only
  ever embedded once per model.

The retriever only needs something that turns text into a vector, which is
the :class:`Embedder` protocol. That keeps the tests independent of whether
a model server happens to be running.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence, runtime_checkable

from rageval.cache import DiskCache


class EmbeddingError(RuntimeError):
    """Raised when an embedding could not be produced."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn text into a vector."""

    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...


class OllamaEmbedder:
    """Calls an Ollama server through its OpenAI compatible endpoint.

    Ollama ignores the API key it is given but insists on one being present,
    so a placeholder is sent unless the caller supplies a real one.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        batch_size: int = 16,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "ollama"
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        # The name identifies the embedding source in cache keys, so it has
        # to carry the model: two models must never share a cached vector.
        self.name = f"ollama:{model}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding per text."""
        items = list(texts)
        if not items:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            vectors.extend(self._embed_batch(items[start : start + self.batch_size]))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise EmbeddingError(
                f"Ollama rejected the embedding request for model "
                f"'{self.model}' (HTTP {error.code}). {detail.strip()}"
            ) from error
        except urllib.error.URLError as error:
            raise EmbeddingError(
                f"Could not reach Ollama at {self.base_url}. Is it running? "
                "Start it with 'ollama serve' and make sure the model is "
                f"available with 'ollama pull {self.model}'."
            ) from error
        except TimeoutError as error:
            raise EmbeddingError(
                f"Ollama did not respond within {self.timeout:g} seconds while "
                f"embedding with model '{self.model}'. The model may still be "
                "loading into memory."
            ) from error

        return self._parse(body, len(batch))

    def _parse(self, body: dict, expected: int) -> list[list[float]]:
        data = body.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise EmbeddingError(
                f"Ollama returned {len(data) if isinstance(data, list) else 'no'} "
                f"embeddings for {expected} input(s)."
            )

        ordered = sorted(data, key=lambda item: item.get("index", 0))
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding")
            if not isinstance(vector, list) or not vector:
                raise EmbeddingError("Ollama returned an empty embedding.")
            vectors.append([float(value) for value in vector])
        return vectors


class CachedEmbedder:
    """Memoises embeddings on disk, keyed by model and text.

    Embedding a corpus is the expensive part of dense retrieval. The cache
    key covers the exact text and the model name, so a changed corpus or a
    changed model produces a different key rather than a stale vector.
    """

    def __init__(
        self,
        inner: Embedder,
        cache: DiskCache,
        namespace: str = "embeddings",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.name = inner.name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []

        keys = [
            DiskCache.make_key(self.namespace, model=self.inner.name, text=text)
            for text in items
        ]

        vectors: list[list[float] | None] = []
        missing: list[tuple[int, str, str]] = []

        for index, (text, key) in enumerate(zip(items, keys)):
            cached = self.cache.get(key)
            if cached is None:
                vectors.append(None)
                missing.append((index, text, key))
            else:
                vector = cached.get("embedding")
                if not isinstance(vector, list) or not vector:
                    vectors.append(None)
                    missing.append((index, text, key))
                else:
                    vectors.append([float(value) for value in vector])

        if missing:
            computed = self.inner.embed([text for _, text, _ in missing])
            if len(computed) != len(missing):
                raise EmbeddingError(
                    f"Expected {len(missing)} embeddings but got {len(computed)}."
                )
            for (index, _, key), vector in zip(missing, computed):
                vectors[index] = vector
                self.cache.set(key, {"embedding": vector})

        return [vector for vector in vectors if vector is not None]
