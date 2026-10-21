"""A judge that talks to a local model served by Ollama.

This is the fallback for when there is no hosted judge available, and it is
the one that gets validated in this repository, because a local judge needs
no account and no key.

It uses Ollama's native chat endpoint at ``/api/chat`` rather than the
OpenAI compatible one at ``/v1/chat/completions``, for the same reason the
generator does: only the native endpoint honours the ``think`` parameter,
and a judge that spends a hundred tokens reasoning about a one line answer
is paying for nothing.

The judge should be a different model family from the generator. A judge
that shares a family with the generator tends to score that generator's
phrasing higher than it deserves, and the agreement figures then measure
self-preference rather than correctness. The framework cannot enforce this
from the outside, so it says so in the name of the judge and in the README.

The API key is read from the environment by the caller. Ollama ignores it,
but it insists on one being present, so a placeholder is sent when none is
configured.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from rageval.judge.base import JudgeError


class OllamaJudge:
    """Grading calls against a local model served by Ollama."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
        timeout: float = 180.0,
        think: bool | None = None,
        json_mode: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "ollama"
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Only some models have a reasoning mode, so the parameter is sent
        # only when it is set.
        self.think = think
        self.json_mode = json_mode
        self.name = f"ollama:{model}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def cache_key(self) -> dict[str, Any]:
        """Everything that changes the verdict goes into the cache key."""
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "think": self.think,
            "json_mode": self.json_mode,
        }

    def build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the request body for one grading call."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.max_tokens,
            },
        }
        if self.json_mode:
            payload["format"] = "json"
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send one chat request and return the assistant's reply."""
        payload = json.dumps(self.build_payload(messages)).encode("utf-8")
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
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise JudgeError(
                f"Ollama rejected the grading request for model "
                f"'{self.model}' (HTTP {error.code}). {detail.strip()}"
            ) from error
        except TimeoutError as error:
            raise JudgeError(
                f"Ollama did not respond within {self.timeout:g} seconds while "
                f"grading with model '{self.model}'. The model may still be "
                "loading into memory."
            ) from error
        except urllib.error.URLError as error:
            raise JudgeError(
                f"Could not reach Ollama at {self.base_url}. Is it running? "
                "Start it with 'ollama serve' and make sure the model is "
                f"available with 'ollama pull {self.model}'."
            ) from error
        except OSError as error:
            raise JudgeError(
                f"The request to Ollama at {self.base_url} failed: {error}"
            ) from error

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as error:
            raise JudgeError(
                "Ollama returned a response that was not valid JSON."
            ) from error
        if not isinstance(body, dict):
            raise JudgeError("Ollama returned a response that was not an object.")
        return self._content_of(body)

    def _content_of(self, body: dict[str, Any]) -> str:
        """Pull the assistant's message out of the response."""
        message = body.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "".join(parts)
        return ""
