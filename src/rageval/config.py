"""Run configuration.

A configuration is a plain dataclass tree so it can be serialised into every
report, diffed between runs, and overridden from the command line without
pulling in a dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from rageval import paths


def _build(cls: type, data: dict[str, Any], section: str):
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be an object.")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"Unknown option(s) in '{section}': {', '.join(sorted(unknown))}. "
            f"Valid options: {', '.join(sorted(known))}."
        )
    return cls(**data)


def _coerce(value: str) -> Any:
    """Turn a command line string into the most specific type it fits."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


#: The relevance floor for each retriever, expressed in that retriever's own
#: score scale. The scales differ: the lexical retriever reports the
#: fraction of the question's content words a chunk covers, while the dense
#: retriever reports a cosine similarity.
#:
#: Each value sits in the middle of the window in which the retriever keeps
#: every answerable case while rejecting as many unanswerable ones as it
#: can. The windows are measured, not guessed, and the README lists them:
#:
#: ==========  ==========  ============================================
#: Retriever   Floor       Safe window on the bundled dataset
#: ==========  ==========  ============================================
#: lexical     0.30        0.22 to 0.37
#: dense       0.35        0.29 to 0.40
#: hybrid      0.40        0.32 to 0.46
#: ==========  ==========  ============================================
DEFAULT_MIN_SCORES = {
    "lexical": 0.30,
    "dense": 0.35,
    "hybrid": 0.40,
}


#: The root of a local Ollama server. The language model generator talks to
#: the native chat API under it, because that is the only endpoint that
#: honours the ``think`` parameter.
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

#: Ollama's OpenAI compatible prefix, used for embeddings.
DEFAULT_OLLAMA_OPENAI_URL = "http://localhost:11434/v1"

#: Google's OpenAI compatible prefix for the Gemini API.
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


@dataclass
class RetrievalConfig:
    """How documents are chunked and how relevant chunks are selected."""

    retriever: str = "lexical"
    top_k: int = 3
    chunk_size: int = 50
    chunk_overlap: int = 1
    min_score: float | None = None
    lexical_weight: float = 0.5
    dense_weight: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["min_score"] = self.resolved_min_score
        return data

    @property
    def resolved_min_score(self) -> float:
        """The relevance floor that will actually be applied.

        ``min_score`` is an override. When it is left unset, each retriever
        falls back to the floor that was calibrated for its own score
        scale.
        """
        if self.min_score is not None:
            return self.min_score
        return DEFAULT_MIN_SCORES.get(self.retriever, 0.0)


@dataclass
class DenseConfig:
    """Settings for the embedder that backs the dense retriever.

    Ollama ignores the API key it is handed but insists on one being
    present, so ``api_key_env`` is optional and is only needed for a proxy
    that actually checks it. Like everywhere else in this project, the key
    is referenced by environment variable name and never stored.
    """

    model: str = "embeddinggemma"
    base_url: str = DEFAULT_OLLAMA_OPENAI_URL
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    batch_size: int = 16
    query_prefix: str = ""
    document_prefix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    """Which model writes the answer, and how.

    API keys are referenced by environment variable name rather than stored
    here, so that a configuration file can be committed safely.

    ``min_answer_score`` applies to the extractive generator only. The
    language model generator decides for itself whether to abstain, and a
    guard rail applied around it would measure the guard rail rather than
    the model.

    ``think`` is passed through only when it is set, because it is a
    parameter that only some models accept. Leaving it unset keeps the
    request portable.
    """

    provider: str = "extractive"
    model: str = "extractive"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 256
    base_url: str | None = None
    api_key_env: str | None = None
    min_answer_score: float = 0.45
    think: bool | None = None
    timeout_seconds: float = 180.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeConfig:
    """LLM-as-judge settings.

    The judge must be a different model family from the generator, otherwise
    it tends to score its own phrasing higher than it deserves. The
    framework cannot enforce that from the outside, so it is worth checking
    by hand when the judge is set up.

    ``threshold`` is the correctness score at which a verdict counts as
    correct. The judge is asked for the four rubric scores only, and the
    binary verdict is derived from that one score, so that the framework
    never has two fields that can contradict each other.

    ``n_samples`` and ``swap_order`` exist to make the judge's own
    behaviour visible. A judge that changes its mind about the same answer
    between two calls, or when the reference and the graded answer swap
    places, is measuring something other than the answer.
    """

    enabled: bool = False
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    base_url: str = ""
    api_key_env: str = "GEMINI_API_KEY"
    n_samples: int = 3
    swap_order: bool = True
    temperature: float = 0.0
    max_tokens: int = 512
    threshold: int = 4
    seed: int = 42
    think: bool | None = None
    timeout_seconds: float = 120.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["base_url"] = self.resolved_base_url
        return data

    @property
    def resolved_base_url(self) -> str:
        """The endpoint the judge will actually be called on.

        The default depends on the provider, so a judge pointed at Gemini
        and a judge pointed at a local Ollama server do not need a base URL
        written down to be distinguishable.
        """
        if self.base_url:
            return self.base_url
        if self.provider == "ollama":
            return DEFAULT_OLLAMA_BASE_URL
        return DEFAULT_GEMINI_BASE_URL


@dataclass
class EvalConfig:
    """Everything needed to reproduce one evaluation run."""

    dataset_path: Path = paths.DEFAULT_DATASET_PATH
    knowledge_base_path: Path = paths.DEFAULT_KNOWLEDGE_BASE_PATH
    reports_dir: Path = paths.REPORTS_DIR
    cache_dir: Path = paths.CACHE_DIR
    run_name: str = ""
    use_cache: bool = True
    correct_recall_threshold: float = 0.9
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dense: DenseConfig = field(default_factory=DenseConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": str(self.dataset_path),
            "knowledge_base_path": str(self.knowledge_base_path),
            "reports_dir": str(self.reports_dir),
            "cache_dir": str(self.cache_dir),
            "run_name": self.run_name,
            "use_cache": self.use_cache,
            "correct_recall_threshold": self.correct_recall_threshold,
            "retrieval": self.retrieval.to_dict(),
            "dense": self.dense.to_dict(),
            "generation": self.generation.to_dict(),
            "judge": self.judge.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object.")

        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown config option(s): {', '.join(sorted(unknown))}. "
                f"Valid options: {', '.join(sorted(known))}."
            )

        payload = {
            key: value
            for key, value in data.items()
            if key not in ("retrieval", "dense", "generation", "judge")
        }
        for key in ("dataset_path", "knowledge_base_path", "reports_dir", "cache_dir"):
            if key in payload:
                payload[key] = Path(payload[key]).expanduser()

        config = cls(**payload)
        if "retrieval" in data:
            config.retrieval = _build(RetrievalConfig, data["retrieval"], "retrieval")
        if "dense" in data:
            config.dense = _build(DenseConfig, data["dense"], "dense")
        if "generation" in data:
            config.generation = _build(GenerationConfig, data["generation"], "generation")
        if "judge" in data:
            config.judge = _build(JudgeConfig, data["judge"], "judge")
        return config

    @classmethod
    def load(cls, path: Path | None = None) -> EvalConfig:
        """Load a configuration from a JSON file, or return the defaults."""
        if path is None:
            return cls()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def apply_overrides(self, overrides: list[str]) -> EvalConfig:
        """Apply dotted ``section.field=value`` overrides in place."""
        for raw in overrides:
            if "=" not in raw:
                raise ValueError(
                    f"Invalid override '{raw}'. Expected the form section.field=value."
                )
            dotted, _, value = raw.partition("=")
            self._set_dotted(dotted.strip(), _coerce(value))
        return self

    def _set_dotted(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        if len(parts) == 1:
            target = self
            name = parts[0]
        elif len(parts) == 2:
            section = getattr(self, parts[0], None)
            if section is None or not is_dataclass(section):
                raise ValueError(f"Unknown config section '{parts[0]}'.")
            target = section
            name = parts[1]
        else:
            raise ValueError(
                f"Invalid override path '{dotted}'. Expected section.field."
            )

        # Only real fields can be set. Checking against the dataclass fields
        # rather than with hasattr keeps read-only properties, such as the
        # resolved relevance floor, out of reach.
        known = {f.name for f in fields(target)}
        if name not in known:
            raise ValueError(f"Unknown config option '{dotted}'.")
        setattr(target, name, value)
