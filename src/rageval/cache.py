"""A small on-disk cache for model responses.

Evaluation runs repeat the same prompts constantly: once while iterating on a
metric, again while iterating on the prompt, and again when generating a
report. Caching turns those repeats into free, deterministic lookups, which
matters a great deal when the upstream API has a per-day request cap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


class DiskCache:
    """Stores JSON payloads under a directory, keyed by a content hash."""

    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled = enabled
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(namespace: str, **parts: Any) -> str:
        """Build a stable cache key from a namespace and arbitrary parts."""
        payload = json.dumps(
            {"namespace": namespace, **parts},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _path_for(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

    def get_or_compute(
        self,
        key: str,
        compute: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Return ``(value, was_cached)``, computing on a miss."""
        cached = self.get(key)
        if cached is not None:
            return cached, True
        value = compute()
        self.set(key, value)
        return value, False
