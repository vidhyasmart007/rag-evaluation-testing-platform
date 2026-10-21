"""A generator wrapper that memoises answers on disk.

Evaluation runs repeat the same prompts constantly: once while iterating on
a metric, again while iterating on the prompt, and again when a report is
regenerated. Serving those repeats from disk turns them into free lookups,
which matters a great deal once the generator is a hosted model with a
per-day request cap.

The cache key covers the question, the retrieved context and the model
name, so a change to the retriever invalidates the entry automatically.
The package fingerprint is part of the namespace, so a change to the
code invalidates it too.
"""

from __future__ import annotations

from rageval.cache import DiskCache
from rageval.generation.base import GenerationResult, Generator


class CachedGenerator:
    """Wraps a generator and serves repeated answers from the disk cache."""

    def __init__(
        self,
        inner: Generator,
        cache: DiskCache,
        namespace: str = "generation",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.name = inner.name

    def generate(self, question: str, context: str) -> GenerationResult:
        key = DiskCache.make_key(
            self.namespace,
            model=self.inner.name,
            question=question,
            context=context,
        )

        cached = self.cache.get(key)
        if cached is not None:
            return GenerationResult(
                text=cached.get("text", ""),
                model=cached.get("model", self.inner.name),
                prompt_tokens=int(cached.get("prompt_tokens", 0)),
                completion_tokens=int(cached.get("completion_tokens", 0)),
                cached=True,
                abstained=bool(cached.get("abstained", False)),
            )

        result = self.inner.generate(question, context)
        self.cache.set(
            key,
            {
                "text": result.text,
                "model": result.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "abstained": result.abstained,
            },
        )
        return result
