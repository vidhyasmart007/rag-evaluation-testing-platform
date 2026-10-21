"""A judge that talks to an OpenAI compatible chat endpoint.

Google's Gemini API is reachable through an OpenAI compatible prefix at
``https://generativelanguage.googleapis.com/v1beta/openai/``, and Ollama
serves one at ``http://localhost:11434/v1``. The same client therefore
covers both, and the same rubric runs unchanged against either of them.

The API key is read from the environment by the caller and is never stored
in a configuration file, so a run can be reproduced from its report without
leaking a secret. The key is sent as a bearer token because that is what
the endpoint expects, and Ollama ignores it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from rageval.judge.base import JudgeError


class OpenAICompatJudge:
    """Grading calls against an OpenAI compatible chat completion endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        timeout: float = 120.0,
        json_mode: bool = True,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout = timeout
        self.json_mode = json_mode
        self.name = f"openai-compat:{model}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def cache_key(self) -> dict[str, Any]:
        """Everything that changes the verdict goes into the cache key."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "json_mode": self.json_mode,
        }

    def build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the request body for one grading call."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.json_mode:
            # Constraining the output to JSON is stronger than asking for
            # JSON in the prompt, which is all a small model really does.
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Send one chat request and return the assistant's reply."""
        payload = json.dumps(self.build_payload(messages)).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            method="POST",
            headers=headers,
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise JudgeError(
                f"The judge endpoint rejected the request for model "
                f"'{self.model}' (HTTP {error.code}). {detail.strip()}"
            ) from error
        except TimeoutError as error:
            raise JudgeError(
                f"The judge at {self.base_url} did not respond within "
                f"{self.timeout:g} seconds."
            ) from error
        except urllib.error.URLError as error:
            raise JudgeError(
                f"Could not reach the judge at {self.base_url}: {error.reason}"
            ) from error
        except OSError as error:
            raise JudgeError(
                f"The request to the judge at {self.base_url} failed: {error}"
            ) from error

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as error:
            raise JudgeError(
                "The judge endpoint returned a response that was not valid JSON."
            ) from error
        if not isinstance(body, dict):
            raise JudgeError(
                "The judge endpoint returned a response that was not an object."
            )
        return self._content_of(body)

    def _content_of(self, body: dict[str, Any]) -> str:
        """Pull the assistant's message out of the response."""
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, dict):
            return ""
        message = first.get("message")
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
