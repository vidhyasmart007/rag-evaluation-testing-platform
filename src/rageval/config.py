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


@dataclass
class RetrievalConfig:
    """How documents are chunked and how relevant chunks are selected."""

    retriever: str = "lexical"
    top_k: int = 3
    chunk_size: int = 50
    chunk_overlap: int = 1
    min_score: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationConfig:
    """Which model writes the answer, and how.

    API keys are referenced by environment variable name rather than stored
    here, so that a configuration file can be committed safely.
    """

    provider: str = "extractive"
    model: str = "extractive"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 256
    base_url: str | None = None
    api_key_env: str | None = None
    min_answer_score: float = 0.45

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeConfig:
    """LLM-as-judge settings.

    The judge must be a different model family from the generator, otherwise
    it tends to score its own phrasing higher than it deserves.
    """

    enabled: bool = False
    provider: str = "gemini"
    model: str = "gemini-3.8-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env: str = "GEMINI_API_KEY"
    n_samples: int = 3
    swap_order: bool = True
    temperature: float = 0.0
    max_tokens: int = 512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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

        payload = {key: value for key, value in data.items() if key not in ("retrieval", "generation", "judge")}
        for key in ("dataset_path", "knowledge_base_path", "reports_dir", "cache_dir"):
            if key in payload:
                payload[key] = Path(payload[key]).expanduser()

        config = cls(**payload)
        if "retrieval" in data:
            config.retrieval = _build(RetrievalConfig, data["retrieval"], "retrieval")
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

        if not hasattr(target, name):
            raise ValueError(f"Unknown config option '{dotted}'.")
        setattr(target, name, value)
