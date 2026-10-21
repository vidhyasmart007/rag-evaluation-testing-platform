# RAG Evaluation & Testing Platform

A framework for measuring whether a Retrieval-Augmented Generation (RAG) pipeline
actually works — not whether it *looks* like it works.

It runs a fixed dataset of questions against a knowledge base, captures what the
retriever returned and what the generator answered, scores both, and writes a
report you can diff between runs. The point is to make regressions visible and
to expose the failure modes that a demo hides.

```text
question ──▶ retriever ──▶ chunks ──▶ generator ──▶ answer
                 │                        │              │
                 └── retrieval metrics ──┴── answer metrics
                                                    │
                                       report (JSON + CSV)
```

---

## Quick start

```bash
git clone https://github.com/vidhyasmart007/rag-evaluation-testing-platform.git
cd rag-evaluation-testing-platform

python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest          # the only runtime dependency is pytest

PYTHONPATH=src python -m rageval.cli run
```

`PYTHONPATH=src` is the primary way to run this project. The package uses a `src`
layout, so the importable package lives in `src/rageval`. Setting `PYTHONPATH`
means every command works with no install step at all.

If you would rather install it:

```bash
pip install -e ".[dev]"     # adds the `rag-eval` console script
rag-eval run
```

A `Makefile` wraps the common commands and always sets `PYTHONPATH` for you:

```bash
make run       # run the evaluation suite
make test      # run the test suite
make cases     # list the dataset cases
make config    # print the effective configuration
make retrieve QUESTION="Where is the office?"
make install   # editable install, with a macOS fix (see Troubleshooting)
```

---

## Project layout

```text
.
├── data/
│   ├── knowledge_base.txt      # the source document (16 sentences)
│   └── eval_dataset.json       # 21 evaluation cases, schema v2
├── reports/
│   ├── latest.json             # full report of the most recent run
│   └── latest.csv              # the same run, one row per case
├── src/rageval/
│   ├── paths.py                # every path resolved from the repo root
│   ├── text.py                 # tokenizer, stemmer, IDF weights, coverage
│   ├── schema.py               # TestCase, RetrievedChunk, CaseResult, RunResult
│   ├── config.py               # EvalConfig and dotted --set overrides
│   ├── dataset.py              # dataset loading and validation
│   ├── cache.py                # on-disk response cache
│   ├── runner.py               # the evaluation loop — all the logic lives here
│   ├── reporting.py            # JSON + CSV writers
│   ├── cli.py                  # thin command line wrapper
│   ├── retrieval/
│   │   ├── base.py             # Retriever protocol
│   │   └── lexical.py          # sentence splitting, chunking, BM25
│   ├── generation/
│   │   ├── base.py             # Generator protocol
│   │   └── extractive.py       # baseline generator
│   └── metrics/
│       ├── answer.py           # token_f1, exact_match, faithfulness, ...
│       └── retrieval.py        # hit@k, recall@k, MRR
├── tests/                      # 103 tests
├── Makefile
└── pyproject.toml
```

The important structural decision: **all evaluation logic lives in `runner.py`**.
The CLI, the HTTP API and the dashboard are thin wrappers around
`run_evaluation(config)`. When the UI arrives it will call the same function the
command line calls, so a score in the dashboard and a score in a CSV can never
disagree.

---

## The dataset format (schema v2)

```json
{
  "version": "2.0.0",
  "description": "Questions answered by the company knowledge base.",
  "cases": [
    {
      "id": 2,
      "question": "How many annual leave days do employees receive?",
      "reference_answer": "Employees receive 18 annual leave days per calendar year.",
      "answerable": true,
      "relevant_sentence_ids": [3],
      "difficulty": "easy",
      "tags": ["policy"]
    },
    {
      "id": 17,
      "question": "How many sick leave days do employees receive?",
      "reference_answer": "",
      "answerable": false,
      "relevant_sentence_ids": [],
      "difficulty": "hard",
      "tags": ["unanswerable", "partial-overlap"]
    }
  ]
}
```

Two design points worth calling out:

- **`relevant_sentence_ids` points at sentence ids, not chunk ids.** Chunk
  boundaries move whenever `chunk_size` changes. Labels pinned to chunk ids
  would rot silently; sentence ids do not.
- **There is no `actual_answer` field.** Answers must be produced by the
  pipeline under test. A hand-written answer in the dataset is a lie about what
  the system does. The old format had one; it was removed, and the legacy
  loader silently ignores it.

`answerable: false` marks a question the knowledge base cannot answer. Those
cases exist so the framework can score **abstention** — whether the system
correctly refuses — instead of only scoring answers.

The loader also accepts the legacy flat list format and reports a warning.

---

## Metrics

| Metric | What it measures | Range |
| --- | --- | --- |
| `hit_at_k` | A retrieved chunk contains a labelled sentence | 0–1 |
| `recall_at_k` | Fraction of labelled sentences that were retrieved | 0–1 |
| `reciprocal_rank` | How high the first correct chunk ranked | 0–1 |
| `retrieval_relevance` | Best chunk's IDF-weighted question coverage | 0–1 |
| `answer_relevance` | Question/answer token overlap | 0–1 |
| `faithfulness` | Fraction of answer tokens supported by the context | 0–1 |
| `exact_match` | Answer tokens equal the reference tokens | 0–1 |
| `token_recall` | Fraction of reference tokens present in the answer | 0–1 |
| `token_f1` | Harmonic mean of precision and recall over tokens | 0–1 |
| `numeric_match` | Numbers in the answer match the reference | 0–1 |
| `answer_correct` | The headline answer score | 0–1 |
| `abstention_correct` | Answered iff the case was answerable | 0–1 |
| `unanswerable_abstention_rate` | Of the unanswerable cases, how many were refused | 0–1 |
| `hallucination_rate` | Of the unanswerable cases, how many were answered anyway | 0–1 |

`answer_correct` is one documented rule: exact match **or** numeric match **or**
token recall at or above `correct_recall_threshold` (default `0.9`).

Metrics that do not apply to a case are **omitted, not zeroed**. An unanswerable
case has no reference answer, so it has no `token_f1`; recording a zero there
would drag the average down for a metric that was never measured. Aggregates
average only the cases that were actually measured.

### Why the tokenizer is shared

`src/rageval/text.py` is the single source of truth for how words are compared.
Retrieval ranking, retrieval relevance, answer relevance and faithfulness all
run on the same stems. Two halves of a framework that tokenize differently will
drift apart, and the disagreement shows up as a mysterious score.

The stemmer is deliberately shallow. It never shortens a word below three
characters, and it drops a trailing silent `e` both before and after stripping
a suffix, so `measure`/`measures` and `employee`/`employees` agree.

---

## Commands

```bash
PYTHONPATH=src python -m rageval.cli run
PYTHONPATH=src python -m rageval.cli cases
PYTHONPATH=src python -m rageval.cli config
PYTHONPATH=src python -m rageval.cli retrieve "How many leave days do employees get?"
```

| Command | What it does |
| --- | --- |
| `run` | Evaluate the whole dataset and write a report |
| `cases` | List the dataset cases and any loader warnings |
| `config` | Print the effective configuration as JSON |
| `retrieve` | Show what the retriever returns for one question |

Useful flags on `run`:

```bash
--run-name my_run        # name the run (default: run_<timestamp>)
--dataset path.json      # use a different dataset
--no-cache               # bypass the response cache
--json                   # print the raw report to stdout
--set retrieval.top_k=5  # override any configuration value
```

Each run writes `reports/<run_id>.json` and `reports/<run_id>.csv`, and refreshes
`reports/latest.json` and `reports/latest.csv`. The JSON report contains the
full configuration, the dataset version and content hash, the git commit, the
per-case metrics and the aggregates — enough to reproduce and to diff.

### Caching

Generated answers are cached on disk under `.cache/`, keyed by the question,
the retrieved context and the model name. Re-running the same evaluation
serves every answer from disk instead of regenerating it, which matters once
the generator is a hosted model with a per-day request cap.

The cache key also includes a fingerprint of the package source, so editing
any module invalidates every entry. A stale answer can never be served after
a change to the tokenizer, the metrics or the generator. `--no-cache` turns it
off, and the `cache_hits` aggregate reports how many cases were served from
disk.

---

## Configuration

Configuration is a plain dataclass tree. It can be written as JSON, loaded from
a file, and overridden from the command line with dotted paths.

```bash
PYTHONPATH=src python -m rageval.cli run \
  --set retrieval.top_k=5 \
  --set retrieval.min_score=0.3 \
  --set generation.min_answer_score=0.5
```

```json
{
  "retrieval": {
    "retriever": "lexical",
    "top_k": 3,
    "chunk_size": 50,
    "chunk_overlap": 1,
    "min_score": 0.15
  },
  "generation": {
    "provider": "extractive",
    "model": "extractive",
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": 256,
    "base_url": null,
    "api_key_env": null,
    "min_answer_score": 0.45
  },
  "judge": {
    "enabled": false,
    "provider": "gemini",
    "model": "gemini-3.8-flash",
    "api_key_env": "GEMINI_API_KEY",
    "n_samples": 3,
    "swap_order": true,
    "temperature": 0.0
  }
}
```

**API keys are referenced by environment variable name, never stored.** The
configuration file gets committed, so it must not hold secrets. `api_key_env:
"GEMINI_API_KEY"` means "read the key from `$GEMINI_API_KEY`", and there is a
test that fails if a plain `api_key` field ever appears in the schema.

Unknown keys are rejected rather than ignored, in both files and `--set`
overrides. A typo in a config file should fail loudly.

### What the two thresholds do

- `retrieval.min_score` (0.15) is a **relevance floor**. It trims chunks that
  are clearly noise out of the context. It is not the abstention mechanism.
- `generation.min_answer_score` (0.45) is the **abstention floor**. The
  generator refuses to answer when no sentence in the context covers enough of
  the question's content words.

These are separated on purpose. Trying to make one threshold do both jobs
produces a number that is hard to reason about and hard to tune.

---

## Current baseline

Lexical retriever (BM25) plus the extractive generator, over 21 cases
(17 answerable, 4 unanswerable):

| Metric | Value |
| --- | --- |
| `hit_at_k` | **1.0000** |
| `recall_at_k` | **1.0000** |
| `reciprocal_rank` | **1.0000** |
| `answer_correct` | **0.9412** (16 / 17) |
| `token_recall` | 0.9765 |
| `numeric_match` | 1.0000 |
| `unanswerable_abstention_rate` | **0.7500** (3 / 4) |
| `hallucination_rate` | 0.2500 |
| `abstention_correct` | 0.9524 |
| `retrieval_relevance` | 0.7541 |
| `faithfulness` | 0.8571 |
| `error_count` | 0 |

Reproduce it with `make run`; the numbers land in `reports/latest.json`.

---

## Known limitations

These are real, measured, and deliberately left in place. Fixing them is what
the later phases are for.

**1. One unanswerable case cannot be rejected by lexical retrieval.**
Case 17 asks *"How many sick leave days do employees receive?"*. The word
`sick` appears nowhere in the knowledge base, but `leave`, `days`, `employees`
and `receive` all do — so the retriever confidently returns the annual leave
sentence and the generator answers "18". Distinguishing *"how many leave days"*
from *"how many sick leave days"* requires understanding that `sick` modifies
the answer, not understanding vocabulary alone. No threshold fixes this: the
question's score is 0.80 and the rejected case's score is 0.40, so any
threshold that rejects one rejects the other. This is the clearest argument for
moving to a language model generator.

**2. The extractive baseline can only return one sentence.**
Case 8 (*"What is Retrieval-Augmented Generation?"*) has a reference answer that
spans two sentences of the source document. A generator that returns exactly
one sentence scores `token_recall = 0.6` and is marked incorrect. The answer is
not wrong, it is truncated.

**3. Faithfulness is lexical, and lexical faithfulness is nearly circular here.**
The metric asks what fraction of the answer's tokens appear in the context. The
extractive generator builds its answer *out of* the context by copying, so the
metric almost always returns 1.0. It only drops when the generator abstains and
the answer is empty. It cannot detect a contradiction, a paraphrase that
inverts meaning, or an answer that uses the right words for the wrong reason.
A claim-decomposition version that asks "does the context support this
statement?" is the fix.

**4. Retrieval is lexical only.** No embeddings, no reranking. Questions whose
vocabulary does not appear in the corpus are invisible to it.

**5. The answer correctness rule is a proxy.** `token_recall >= 0.9` is a
reasonable stand-in for "says the same thing", but it is not understanding.

Two dataset questions were rewritten during calibration because the framework
surfaced genuine ambiguity in them — the original wording collided with an
unrelated sentence in the corpus. Those rewrites are recorded in the git
history, and they are a fair illustration of what a test set goes through.

---

## Roadmap

| Phase | What lands |
| --- | --- |
| **0** ✅ | Bug fixes, package layout, dataset schema v2, runner, config, cache, tests, docs |
| **1** ✅ | Real metrics, threshold calibration, reproducible baseline report |
| **2** | Dense retrieval (Ollama embeddings) and a hybrid retriever, with the same `Retriever` interface |
| **3** | LLM answer generation via Ollama, seeded for reproducibility, with the extractive baseline kept as a comparison arm |
| **4** | LLM-as-judge on a different model family, validated against hand labels with Cohen's κ before it is trusted |
| **5** | Streamlit dashboard, then a public deployment |

The dashboard comes last on purpose. A UI over broken metrics is a UI over
broken metrics.

---

## Project evolution

The first version of this repository was a flat prototype: `Day1.py`,
`day2.py` and three modules under `src/`. It worked, but it had problems that
made it impossible to extend — the two scripts duplicated the pipeline, the
answers were graded against a hand-written `actual_answer` field in the dataset,
the chunks cut sentences in half, faithfulness compared an answer to a context
the answer had been copied from, and the knowledge base and the dataset
disagreed about how many annual leave days employees get (18 versus 20).

All of those are fixed. The old scripts and modules were removed rather than
left to rot, and this README replaced the one that described them.

---

## Testing

```bash
make test          # or: python -m pytest
```

103 tests across eight modules. They cover the tokenizer and stemmer, chunking
and BM25 ranking, every metric, dataset validation and the legacy format,
configuration overrides and the no-secrets rule, generator abstention, the
response cache, and the runner end to end including the report files it
writes.

The suite runs with no install step: `pyproject.toml` sets
`pythonpath = ["src"]` for pytest.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'rageval'`**

The package lives in `src/rageval`. Either prefix commands with
`PYTHONPATH=src` (or use `make`), or install it with `pip install -e ".[dev]"`.

**The editable install does not take effect on macOS**

macOS propagates the *hidden* file attribute to everything created inside a
hidden directory, and Python 3.13 skips any `.pth` file that carries it. If
your virtual environment is hidden in Finder, every editable install in it
breaks silently. `make install` clears the attribute after installing. The
`PYTHONPATH=src` path is immune to this and is the recommended way to work.

---

## Author

Vidhya A
