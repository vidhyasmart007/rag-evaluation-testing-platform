"""The evaluation runner.

This is the piece the project was missing. Everything else in the package
produces or scores one part of a single case; the runner is what ties them
together and produces a report for the whole dataset.

Keeping the runner free of any web framework means the same function backs
the command line, the HTTP API and the dashboard.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rageval.cache import DiskCache, framework_fingerprint
from rageval.config import DEFAULT_OLLAMA_BASE_URL, EvalConfig
from rageval.dataset import load_dataset
from rageval.generation.base import GenerationResult, Generator
from rageval.generation.cached import CachedGenerator
from rageval.generation.extractive import ExtractiveGenerator
from rageval.generation.ollama import OllamaGenerator
from rageval.metrics.answer import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
    exact_match,
    numeric_match,
    token_f1,
    token_recall,
)
from rageval.metrics.retrieval import (
    evaluate_retrieval,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rageval.reporting import write_reports
from rageval.retrieval.base import Retriever
from rageval.retrieval.dense import DenseRetriever
from rageval.retrieval.embeddings import CachedEmbedder, Embedder, OllamaEmbedder
from rageval.retrieval.hybrid import HybridRetriever
from rageval.retrieval.lexical import BM25Retriever, build_chunks, split_into_sentences
from rageval.schema import CaseResult, RetrievedChunk, RunResult, TestCase


def load_knowledge_base(path: Path) -> str:
    """Read the source document the pipeline answers questions from."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")
    return Path(path).read_text(encoding="utf-8")


def build_embedder(config: EvalConfig) -> Embedder:
    """Create the embedder that backs the dense and hybrid retrievers.

    The API key is read from the environment when one is configured. It is
    never written to a configuration file, so a run can be reproduced from
    the report without leaking a secret.
    """
    dense = config.dense

    api_key = None
    if dense.api_key_env:
        api_key = os.environ.get(dense.api_key_env)

    embedder: Embedder = OllamaEmbedder(
        dense.model,
        base_url=dense.base_url,
        api_key=api_key,
        timeout=dense.timeout_seconds,
        batch_size=dense.batch_size,
    )

    if config.use_cache:
        cache = DiskCache(Path(config.cache_dir), enabled=True)
        embedder = CachedEmbedder(embedder, cache, namespace="embeddings")
    return embedder


def build_retriever(config: EvalConfig, document: str) -> Retriever:
    """Create the retriever described by the configuration."""
    sentences = split_into_sentences(document)
    chunks = build_chunks(
        sentences,
        chunk_size=config.retrieval.chunk_size,
        overlap=config.retrieval.chunk_overlap,
    )

    retriever_name = config.retrieval.retriever
    if retriever_name == "lexical":
        return BM25Retriever(chunks)

    if retriever_name not in ("dense", "hybrid"):
        raise ValueError(
            f"Unknown retriever '{retriever_name}'. "
            "Available: lexical, dense, hybrid."
        )

    dense = DenseRetriever(
        chunks,
        build_embedder(config),
        query_prefix=config.dense.query_prefix,
        document_prefix=config.dense.document_prefix,
    )
    if retriever_name == "dense":
        return dense

    return HybridRetriever(
        chunks,
        BM25Retriever(chunks),
        dense,
        lexical_weight=config.retrieval.lexical_weight,
        dense_weight=config.retrieval.dense_weight,
    )


def build_generator(config: EvalConfig) -> Generator:
    """Create the generator described by the configuration."""
    provider = config.generation.provider
    generation = config.generation

    if provider == "extractive":
        generator: Generator = ExtractiveGenerator(
            min_answer_score=generation.min_answer_score
        )
    elif provider == "ollama":
        if not generation.model or generation.model == "extractive":
            raise ValueError(
                "The ollama provider needs a model. Set generation.model to "
                "the name of a model Ollama has, for example 'qwen3:1.7b'."
            )

        api_key = None
        if generation.api_key_env:
            api_key = os.environ.get(generation.api_key_env)

        generator = OllamaGenerator(
            generation.model,
            base_url=generation.base_url or DEFAULT_OLLAMA_BASE_URL,
            api_key=api_key,
            temperature=generation.temperature,
            seed=generation.seed,
            max_tokens=generation.max_tokens,
            timeout=generation.timeout_seconds,
            think=generation.think,
        )
    else:
        raise ValueError(
            f"Unknown generation provider '{provider}'. "
            "Available: extractive, ollama."
        )

    if config.use_cache:
        cache = DiskCache(Path(config.cache_dir), enabled=True)
        generator = CachedGenerator(
            generator,
            cache,
            namespace=f"generation:{framework_fingerprint()}",
        )
    return generator


def compute_metrics(
    case: TestCase,
    chunks: list[RetrievedChunk],
    generated: GenerationResult,
    context: str,
    config: EvalConfig,
) -> dict[str, float]:
    """Score one case.

    Metrics that do not apply to a case are omitted rather than recorded
    as zero, so that averages are not dragged down by cases the metric
    never had a chance to be measured on.
    """
    answer = generated.text
    metrics: dict[str, float] = {
        "retrieval_relevance": round(evaluate_retrieval(chunks), 6),
        "answer_relevance": round(evaluate_answer_relevance(case.question, answer), 6),
        "faithfulness": round(evaluate_faithfulness(answer, context), 6),
        "abstained": 1.0 if generated.abstained else 0.0,
        # Whether the retriever refused to answer at all. This is a
        # retrieval level decision, made before the generator sees
        # anything, and it is the first place a question the knowledge
        # base cannot answer should be turned away.
        "retrieval_rejected": 1.0 if not chunks else 0.0,
    }

    if case.relevant_sentence_ids:
        metrics["hit_at_k"] = hit_at_k(chunks, case.relevant_sentence_ids)
        metrics["recall_at_k"] = round(
            recall_at_k(chunks, case.relevant_sentence_ids), 6
        )
        metrics["reciprocal_rank"] = round(
            reciprocal_rank(chunks, case.relevant_sentence_ids), 6
        )
        metrics["precision_at_k"] = round(
            precision_at_k(chunks, case.relevant_sentence_ids), 6
        )

    if case.reference_answer:
        exact = exact_match(answer, case.reference_answer)
        recall = token_recall(answer, case.reference_answer)
        metrics["exact_match"] = exact
        metrics["token_f1"] = round(token_f1(answer, case.reference_answer), 6)
        metrics["token_recall"] = round(recall, 6)

        numeric = numeric_match(answer, case.reference_answer)
        if numeric is not None:
            metrics["numeric_match"] = numeric

        # A single documented rule for "correct". Exact match and numeric
        # match are the strict cases; token recall covers answers that are
        # worded differently but say the same thing.
        is_correct = (
            exact >= 1.0
            or numeric == 1.0
            or recall >= config.correct_recall_threshold
        )
        metrics["answer_correct"] = 1.0 if is_correct else 0.0

    if case.answerable:
        metrics["abstention_correct"] = 0.0 if generated.abstained else 1.0
    else:
        metrics["abstention_correct"] = 1.0 if generated.abstained else 0.0

    return metrics


def evaluate_case(
    case: TestCase,
    retriever: Retriever,
    generator: Generator,
    config: EvalConfig,
) -> CaseResult:
    """Run one case through the pipeline and score the outcome."""
    started = perf_counter()

    try:
        chunks = retriever.retrieve(
            case.question,
            config.retrieval.top_k,
            config.retrieval.resolved_min_score,
        )
        context = "\n".join(chunk.text for chunk in chunks)
        generated = generator.generate(case.question, context)
        latency_ms = int((perf_counter() - started) * 1000)

        return CaseResult(
            test_id=case.id,
            question=case.question,
            reference_answer=case.reference_answer,
            answerable=case.answerable,
            retrieved=chunks,
            answer=generated.text,
            abstained=generated.abstained,
            abstention_reason=generated.reason,
            metrics=compute_metrics(case, chunks, generated, context, config),
            latency_ms=latency_ms,
            generator=generated.model,
            cached=generated.cached,
        )
    except Exception as error:  # noqa: BLE001 - one bad case must not kill a run
        latency_ms = int((perf_counter() - started) * 1000)
        return CaseResult(
            test_id=case.id,
            question=case.question,
            reference_answer=case.reference_answer,
            answerable=case.answerable,
            answer="",
            abstained=False,
            metrics={},
            latency_ms=latency_ms,
            generator=getattr(generator, "name", "unknown"),
            error=f"{type(error).__name__}: {error}",
        )


def aggregate(cases: list[CaseResult]) -> dict[str, float]:
    """Average every metric that was measured, plus the run level counters."""
    aggregates: dict[str, float] = {}

    names = sorted({name for case in cases for name in case.metrics})
    for name in names:
        values = [case.metrics[name] for case in cases if name in case.metrics]
        aggregates[name] = round(sum(values) / len(values), 4) if values else 0.0

    answerable = [case for case in cases if case.answerable]
    unanswerable = [case for case in cases if not case.answerable]

    aggregates["total_cases"] = float(len(cases))
    aggregates["answerable_cases"] = float(len(answerable))
    aggregates["unanswerable_cases"] = float(len(unanswerable))
    aggregates["error_count"] = float(sum(1 for case in cases if case.error))
    aggregates["cache_hits"] = float(sum(1 for case in cases if case.cached))

    if cases:
        aggregates["abstention_rate"] = round(
            sum(1 for case in cases if case.abstained) / len(cases), 4
        )
    if unanswerable:
        aggregates["unanswerable_abstention_rate"] = round(
            sum(1 for case in unanswerable if case.abstained) / len(unanswerable), 4
        )
        aggregates["hallucination_rate"] = round(
            sum(1 for case in unanswerable if not case.abstained) / len(unanswerable), 4
        )
        aggregates["unanswerable_rejection_rate"] = round(
            sum(
                1
                for case in unanswerable
                if case.metrics.get("retrieval_rejected") == 1.0
            )
            / len(unanswerable),
            4,
        )

    if answerable:
        aggregates["answerable_rejection_rate"] = round(
            sum(
                1
                for case in answerable
                if case.metrics.get("retrieval_rejected") == 1.0
            )
            / len(answerable),
            4,
        )

    latencies = [case.latency_ms for case in cases]
    if latencies:
        aggregates["mean_latency_ms"] = round(sum(latencies) / len(latencies), 2)

    return aggregates


def current_git_commit() -> str | None:
    """The short commit hash of the working tree, if there is one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def run_evaluation(config: EvalConfig) -> RunResult:
    """Evaluate the whole dataset and write the report."""
    if config.judge.enabled:
        raise NotImplementedError(
            "The LLM-as-judge is not implemented yet. Set judge.enabled to false "
            "or wait for the judge phase."
        )

    dataset = load_dataset(Path(config.dataset_path))
    document = load_knowledge_base(Path(config.knowledge_base_path))
    retriever = build_retriever(config, document)
    generator = build_generator(config)

    created_at = datetime.now(timezone.utc)
    run_id = config.run_name or f"run_{created_at.strftime('%Y%m%d_%H%M%S')}"

    cases = [
        evaluate_case(case, retriever, generator, config) for case in dataset.cases
    ]

    run = RunResult(
        run_id=run_id,
        created_at=created_at.isoformat(),
        config=config.to_dict(),
        dataset_version=dataset.version,
        dataset_hash=dataset.content_hash,
        git_commit=current_git_commit(),
        cases=cases,
        aggregates=aggregate(cases),
    )

    write_reports(run, config)
    return run
