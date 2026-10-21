"""A judge client wrapper that memoises verdicts on disk.

Judging is the most expensive part of an evaluation run: a judge is asked
the same question several times per case, and the same case is judged
again every time a report is regenerated. Caching turns those repeats into
free lookups.

The cache key covers the messages, the model name and whatever the client
says changes its own output, so a change to the rubric or to a judge
setting invalidates the entry automatically.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from rageval.cache import DiskCache


class CachedChatClient:
    """Wraps a chat client and serves repeated replies from the disk cache."""

    def __init__(
        self,
        inner: Any,
        cache: DiskCache,
        namespace: str = "judge",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.namespace = namespace
        self.name = inner.name
        self.cache_key: dict[str, Any] = dict(getattr(inner, "cache_key", {}))

    def complete(self, messages: list[dict[str, str]]) -> str:
        digest = hashlib.sha256(
            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:32]
        key = DiskCache.make_key(
            self.namespace,
            model=self.inner.name,
            digest=digest,
            **self.cache_key,
        )

        cached = self.cache.get(key)
        if cached is not None:
            return str(cached.get("text", ""))

        text = self.inner.complete(messages)
        self.cache.set(key, {"text": text})
        return text
