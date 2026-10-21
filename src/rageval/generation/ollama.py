"""Answer generation with a local language model served by Ollama.

Ollama exposes two chat endpoints. This generator uses the native one at
``/api/chat`` rather than the OpenAI compatible one at ``/v1/chat/completions``,
for one concrete reason: only the native endpoint honours the ``think``
parameter, and it is the parameter that decides whether a reasoning model
spends a hundred tokens thinking before it answers.

On a Qwen3 model the difference is not marginal. With the reasoning mode on,
a one sentence answer arrives in about 130 generated tokens and takes three
times as long; with it off the same answer takes 19 tokens. The mode is not
sent unless it is configured, because it is a parameter that only some
models accept.

Everything else the framework needs is here too: a ``seed`` for
reproducibility, a ``temperature`` of zero, and a JSON output mode that
constrains the model to a parsable object rather than hoping it complies.

The model is asked for a small JSON object, ``{"answer": ..., "abstained":
...}``, because a refusal has to be a first class outcome rather than
something inferred from prose. Asking for JSON does not make a small model
reliable, though, so the response is parsed defensively: a strict parse
first, then an embedded object, then a plain text fallback that looks for
refusal wording.

Three decisions are worth spelling out.

* An empty context never reaches the model. If retrieval found nothing,
  there is nothing to ground an answer in, and spending a model call to
  learn that is wasteful.
* The model decides whether to abstain, not the framework. A guard rail
  applied here would measure the guard rail rather than the generator.
* The prompt is a fixed system message. Everything that varies between
  runs lives in the cache key instead.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from time import perf_counter
from typing import Any

from rageval.generation.base import (
    ABSTAINED_ANSWER,
    GenerationError,
    GenerationResult,
)

SYSTEM_PROMPT = (
    "You answer questions about a company's knowledge base using only the "
    "numbered sources you are given.\n"
    "Rules:\n"
    "1. Use only the sources. Do not use outside knowledge and do not guess.\n"
    '2. If the sources do not contain the answer, set "abstained" to true '
    'and leave "answer" as an empty string.\n'
    '3. If the sources do contain the answer, set "abstained" to false and '
    "give the answer in one short sentence, in your own words.\n"
    "4. Reply with a single JSON object and nothing else:\n"
    '{"answer": "the answer", "abstained": false}'
)

#: Phrases that mean the model refused. These are only consulted when the
#: response could not be parsed as JSON, or when a parsed answer still reads
#: like a refusal. They are deliberately blunt: a false positive here turns
#: an honest answer into a refusal, which is the worse error.
REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "don't know",
    "do not know",
    "not enough information",
    "no information",
    "no relevant information",
    "cannot be determined",
    "cannot determine",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not mentioned in",
    "not specified in",
    "not stated in",
    "not provided in",
    "not found in",
    "no answer",
    "unknown",
)

#: Reasons a generator can abstain. They are recorded in the report so that
#: a retrieval failure and a generation failure are never confused.
REASON_EMPTY_CONTEXT = "empty_context"
REASON_MODEL_REFUSAL = "model_refusal"
REASON_EMPTY_ANSWER = "empty_answer"


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response.

    Small models wrap their JSON in prose, fence it in a code block, or
    trail a note after it. All three are common enough that a strict parse
    would throw away most of the answers.
    """
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]

    if stripped.startswith("```"):
        unfenced = stripped.strip("`").strip()
        if unfenced.lower().startswith("json"):
            unfenced = unfenced[4:].strip()
        candidates.append(unfenced)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def looks_like_refusal(text: str) -> bool:
    """Whether a piece of text reads as a refusal to answer."""
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _as_bool(value: Any) -> bool | None:
    """Coerce a model's ``abstained`` field to a boolean, or give up."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0", ""):
            return False
    return None


class OllamaGenerator:
    """Answers a question from retrieved context using a local model.

    The model is addressed through Ollama's own chat API. The API key is
    optional: Ollama ignores it, but a proxy in front of the server might
    not, so one is sent when the configuration names an environment
    variable to read it from.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 256,
        timeout: float = 180.0,
        think: bool | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        # Ollama ignores the key but insists on one being present, so a
        # placeholder is sent unless the caller supplies a real one.
        self.api_key = api_key or "ollama"
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.timeout = timeout
        # Only some models have a reasoning mode. Leaving this unset means
        # the parameter is not sent at all, which is what every other model
        # expects.
        self.think = think
        self.system_prompt = system_prompt
        self.name = f"ollama:{model}"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def cache_key(self) -> dict[str, Any]:
        """Everything that changes the answer goes into the cache key.

        A cached answer produced at one temperature is not an answer to the
        question asked at another, and a change of seed is a change of
        experiment.
        """
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "think": self.think,
        }

    def build_messages(self, question: str, context: str) -> list[dict[str, str]]:
        """Build the chat messages for one question.

        The context is numbered so that the model can refer to a source, and
        so that a reader of the transcript can tell which passage produced
        which part of the answer. Splitting on newlines is safe because a
        chunk is always a single line.
        """
        passages = [line.strip() for line in context.splitlines() if line.strip()]
        if passages:
            sources = "\n".join(
                f"[{index}] {passage}" for index, passage in enumerate(passages, 1)
            )
            user_message = f"Sources:\n{sources}\n\nQuestion: {question}"
        else:
            user_message = f"Question: {question}"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def build_payload(self, question: str, context: str) -> dict[str, Any]:
        """Build the request body for one question."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.build_messages(question, context),
            "stream": False,
            # Constraining the output to JSON is stronger than asking for
            # JSON in the prompt, which is all a small model really does.
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_predict": self.max_tokens,
            },
        }
        if self.think is not None:
            payload["think"] = self.think
        return payload

    def generate(self, question: str, context: str) -> GenerationResult:
        """Answer ``question`` from ``context``."""
        if not context.strip():
            # Nothing was retrieved, so there is nothing to ground an
            # answer in. Asking the model anyway would only produce a
            # refusal, more slowly.
            return GenerationResult(
                text=ABSTAINED_ANSWER,
                model=self.name,
                abstained=True,
                reason=REASON_EMPTY_CONTEXT,
            )

        started = perf_counter()
        body = self._post(question, context)
        latency_ms = int((perf_counter() - started) * 1000)

        content = self._content_of(body)
        answer, abstained, reason = self._interpret(content)

        return GenerationResult(
            text=answer,
            model=self.name,
            prompt_tokens=int(body.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(body.get("eval_count", 0) or 0),
            latency_ms=latency_ms,
            abstained=abstained,
            reason=reason,
        )

    def _content_of(self, body: dict[str, Any]) -> str:
        """Pull the assistant's message out of the response."""
        message = body.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        # Some versions return a list of content parts.
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            return "".join(parts)
        return ""

    def _interpret(self, content: str) -> tuple[str, bool, str]:
        """Turn a model response into an answer, an abstention flag and a reason."""
        payload = extract_json_object(content)
        if payload is not None:
            raw_answer = payload.get("answer")
            answer = raw_answer.strip() if isinstance(raw_answer, str) else ""

            if _as_bool(payload.get("abstained")) is True:
                return ABSTAINED_ANSWER, True, REASON_MODEL_REFUSAL
            if not answer:
                return ABSTAINED_ANSWER, True, REASON_EMPTY_ANSWER
            if looks_like_refusal(answer):
                # The model said it would answer and then refused. The text
                # is the behaviour that matters, so the refusal wins.
                return ABSTAINED_ANSWER, True, REASON_MODEL_REFUSAL
            return answer, False, ""

        # No JSON at all. The model ignored the format, so treat the text as
        # the answer unless it reads like a refusal.
        text = content.strip()
        if not text:
            return ABSTAINED_ANSWER, True, REASON_EMPTY_ANSWER
        if looks_like_refusal(text):
            return ABSTAINED_ANSWER, True, REASON_MODEL_REFUSAL
        return text, False, ""

    def _post(self, question: str, context: str) -> dict[str, Any]:
        """Send one chat request and return the parsed body."""
        payload = json.dumps(self.build_payload(question, context)).encode("utf-8")
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
            raise GenerationError(
                f"Ollama rejected the completion request for model "
                f"'{self.model}' (HTTP {error.code}). {detail.strip()}"
            ) from error
        except TimeoutError as error:
            raise GenerationError(
                f"Ollama did not respond within {self.timeout:g} seconds while "
                f"generating with model '{self.model}'. The model may still be "
                "loading into memory."
            ) from error
        except urllib.error.URLError as error:
            raise GenerationError(
                f"Could not reach Ollama at {self.base_url}. Is it running? "
                "Start it with 'ollama serve' and make sure the model is "
                f"available with 'ollama pull {self.model}'."
            ) from error
        except OSError as error:
            raise GenerationError(
                f"The request to Ollama at {self.base_url} failed: {error}"
            ) from error

        try:
            body = json.loads(raw)
        except json.JSONDecodeError as error:
            raise GenerationError(
                "Ollama returned a response that was not valid JSON."
            ) from error
        if not isinstance(body, dict):
            raise GenerationError("Ollama returned a response that was not an object.")
        return body
