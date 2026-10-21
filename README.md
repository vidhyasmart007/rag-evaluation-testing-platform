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
│   ├── compare.py              # runs all three retrievers and tabulates them
│   ├── retrieval/
│   │   ├── base.py             # Retriever protocol
│   │   ├── lexical.py          # sentence splitting, chunking, BM25
│   │   ├── embeddings.py       # Ollama embeddings client, disk memoisation
│   │   ├── dense.py            # cosine similarity over embedded chunks
│   │   └── hybrid.py           # lexical and dense scores blended
│   ├── generation/
│   │   ├── base.py             # Generator protocol
│   │   └── extractive.py       # baseline generator
│   └── metrics/
│       ├── answer.py           # token_f1, exact_match, faithfulness, ...
│       └── retrieval.py        # hit@k, recall@k, precision@k, MRR
├── tests/                      # 177 tests
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
| `precision_at_k` | Fraction of retrieved chunks that contain something relevant | 0–1 |
| `reciprocal_rank` | How high the first correct chunk ranked | 0–1 |
| `retrieval_relevance` | Best chunk's relevance score, in the retriever's own scale | 0–1 |
| `retrieval_rejected` | The retriever returned nothing above the floor | 0–1 |
| `answer_relevance` | Question/answer token overlap | 0–1 |
| `faithfulness` | Fraction of answer tokens supported by the context | 0–1 |
| `exact_match` | Answer tokens equal the reference tokens | 0–1 |
| `token_recall` | Fraction of reference tokens present in the answer | 0–1 |
| `token_f1` | Harmonic mean of precision and recall over tokens | 0–1 |
| `numeric_match` | Numbers in the answer match the reference | 0–1 |
| `answer_correct` | The headline answer score | 0–1 |
| `abstention_correct` | Answered iff the case was answerable | 0–1 |
| `unanswerable_abstention_rate` | Of the unanswerable cases, how many were refused | 0–1 |
| `unanswerable_rejection_rate` | Of the unanswerable cases, how many were turned away by the retriever | 0–1 |
| `answerable_rejection_rate` | Of the answerable cases, how many the retriever wrongly refused | 0–1 |
| `hallucination_rate` | Of the unanswerable cases, how many were answered anyway | 0–1 |

`answer_correct` is one documented rule: exact match **or** numeric match **or**
token recall at or above `correct_recall_threshold` (default `0.9`).

`precision_at_k` is the metric that says how much **noise** a retriever put
into the context. Recall asks how much of the answer was found; precision asks
how much of the context was irrelevant. For a RAG pipeline the second question
matters, because every irrelevant chunk is another chance for the generator to
answer from the wrong sentence.

`retrieval_rejected` separates two different refusals. The **retriever** turns a
question away when nothing scores above the relevance floor; the **generator**
turns it away when the context does not cover enough of the question. The
`unanswerable_rejection_rate` and `answerable_rejection_rate` aggregates report
the first, `unanswerable_abstention_rate` reports the second.

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
PYTHONPATH=src python -m rageval.compare
```

| Command | What it does |
| --- | --- |
| `run` | Evaluate the whole dataset and write a report |
| `cases` | List the dataset cases and any loader warnings |
| `config` | Print the effective configuration as JSON |
| `retrieve` | Show what the retriever returns for one question |
| `rageval.compare` | Run all three retrievers on the same dataset and print the metrics that differ |

`rageval.compare` writes its reports to a temporary directory, so running a
comparison never clobbers the baseline in `reports/`.

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

Embeddings are cached the same way, but keyed only by the exact text and the
model name — a vector does not depend on this project's code, so editing a
metric should not force the corpus to be embedded again.

---

## Retrievers

Three retrievers share one interface, so they are interchangeable:

```python
retrieve(question, top_k, min_score) -> list[RetrievedChunk]
```

| Retriever | How it scores a chunk | Score scale |
| --- | --- | --- |
| `lexical` | Okapi BM25 for the ranking, IDF-weighted question coverage for the score | 0–1 |
| `dense` | Cosine similarity of the chunk and question embeddings | 0–1, clamped |
| `hybrid` | `lexical_weight × coverage + dense_weight × cosine` | 0–1 |

The **ranking** and the **reported score** are deliberately separate things. BM25
is a better ranking function than raw term overlap, but its score is unbounded
and has no fixed meaning across queries, so it cannot be mixed with another
signal or compared against a threshold. Coverage answers a question a reader can
understand: *how much of this question does this chunk account for?*

The dense score is a cosine similarity clamped to the unit interval. Clamping
keeps it on the same 0–1 scale as the coverage score, which is what makes the
hybrid blend meaningful — a cosine of 0.8 and a coverage of 0.8 can be averaged
without one of them dominating because of its range.

### Embeddings

Embeddings come from Ollama's OpenAI-compatible endpoint at `/v1/embeddings`,
called with the standard library's `urllib`. The project keeps **zero runtime
dependencies**, and one POST with a JSON body is not worth a dependency.

```bash
ollama serve                      # start the server
ollama pull embeddinggemma        # 621 MB, 768 dimensions
```

Vectors are memoised on disk under `.cache/`, keyed by the exact text and the
model name, so a corpus is only embedded once per model and a changed corpus or
a changed model can never serve a stale vector.

```bash
PYTHONPATH=src python -m rageval.cli run --set retrieval.retriever=dense
PYTHONPATH=src python -m rageval.cli run --set retrieval.retriever=hybrid
```

---

## Configuration

Configuration is a plain dataclass tree. It can be written as JSON, loaded from
a file, and overridden from the command line with dotted paths.

```bash
PYTHONPATH=src python -m rageval.cli run \
  --set retrieval.retriever=hybrid \
  --set retrieval.top_k=5 \
  --set generation.min_answer_score=0.5
```

```json
{
  "retrieval": {
    "retriever": "lexical",
    "top_k": 3,
    "chunk_size": 50,
    "chunk_overlap": 1,
    "min_score": null,
    "lexical_weight": 0.5,
    "dense_weight": 0.5
  },
  "dense": {
    "model": "embeddinggemma",
    "base_url": "http://localhost:11434/v1",
    "api_key_env": null,
    "timeout_seconds": 120.0,
    "batch_size": 16,
    "query_prefix": "",
    "document_prefix": ""
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

### The three thresholds

- **`retrieval.min_score`** is a **relevance floor**, applied in the active
  retriever's own score scale. Leaving it `null` uses the floor that was
  calibrated for that retriever; setting it overrides the floor for every
  retriever at once.
- **`generation.min_answer_score`** (0.45) is the **abstention floor**. The
  generator refuses to answer when no sentence in the context covers enough of
  the question's content words.
- **`retrieval.lexical_weight` / `retrieval.dense_weight`** are the blend used
  by the hybrid retriever. They are normalised, so `0.7`/`0.3` and `7`/`3` mean
  the same thing.

The two floors are separated on purpose. Trying to make one threshold do both
jobs produces a number that is hard to reason about and hard to tune.

The per-retriever floors are not guesses. Each was measured: for every retriever
the table below shows the window in which every answerable case still finds its
answer, and the floor sits in the middle of it.

| Retriever | Floor | Safe window | The window is bounded by |
| --- | --- | --- | --- |
| `lexical` | 0.30 | 0.22 – 0.37 | below: case 16 (*"…new London office?"*) scores 0.220; above: case 4 (*"Which city is the office in?"*) scores 0.369 |
| `dense` | 0.35 | 0.29 – 0.40 | below: case 18 (*"…chief executive officer?"*) scores 0.288; above: case 9 (*"What does faithfulness measure?"*) scores 0.396 |
| `hybrid` | 0.40 | 0.32 – 0.46 | below: case 16 scores 0.322; above: case 4 scores 0.462 |

Raising a floor past the top of its window starts throwing away answers that
were findable; lowering it past the bottom stops rejecting the questions that
should be refused. The windows are only about 0.15 wide, and that is worth
knowing before trusting the floors on a new dataset.

---

## Current baseline

The extractive generator over 21 cases (17 answerable, 4 unanswerable), with
each retriever at its own calibrated floor:

| Metric | `lexical` | `dense` | `hybrid` |
| --- | --- | --- | --- |
| `hit_at_k` | **1.0000** | **1.0000** | **1.0000** |
| `recall_at_k` | 1.0000 | 1.0000 | 1.0000 |
| `reciprocal_rank` | 1.0000 | 1.0000 | 1.0000 |
| `precision_at_k` | 0.7843 | 0.7451 | **0.8725** |
| `answer_correct` | 0.9412 | 0.9412 | 0.9412 |
| `token_recall` | 0.9765 | 0.9765 | 0.9765 |
| `unanswerable_abstention_rate` | 0.7500 | 0.7500 | 0.7500 |
| `unanswerable_rejection_rate` | **0.7500** | 0.5000 | **0.7500** |
| `answerable_rejection_rate` | 0.0000 | 0.0000 | 0.0000 |
| `abstention_correct` | 0.9524 | 0.9524 | 0.9524 |
| `hallucination_rate` | 0.2500 | 0.2500 | 0.2500 |
| `error_count` | 0 | 0 | 0 |

Reproduce it with `make run`; the numbers land in `reports/latest.json`.
`retrieval_relevance` is deliberately left out of the table: it is a raw score
in the active retriever's scale, so 0.74 for a lexical coverage and 0.47 for a
cosine similarity are not the same measurement.

### What the comparison actually shows

**The hybrid retriever wins on context precision.** At `0.8725` it puts about 11%
less irrelevant text in front of the generator than lexical does, and about 13%
less than dense. It matches lexical on turning unanswerable questions away
(3 of 4) and beats dense (2 of 4), while refusing nothing that was answerable.

**Dense retrieval on its own is the weakest of the three.** It is the only
retriever that refuses *fewer* unanswerable questions than lexical. The reason
is specific: it scores *"When does the company open its new London office?"*
(0.425) **higher** than lexical does (0.220), and above the answerable
*"What does faithfulness measure?"* (0.396). London is not in the corpus, but
"office" and "company" are, and an embedding of that question still lands close
to the office sentence. Lexical at least gets the vocabulary wrong in a way
that punishes it. Dense retrieval is a poor judge of *absence* — it measures how
close two texts are, and an unanswerable question can be very close to a
plausible-sounding sentence.

**The end-to-end answers are identical across all three retrievers.** Every
metric about the *answer* is unchanged. That is not a rounding artefact: the
retriever picks a different top chunk in 4 of the 21 cases, but the extractive
generator then picks the same sentence out of the context either way, because
the chunks overlap by a sentence and the generator chooses on word coverage.

This is the most useful thing Phase 2 found, and it is a result about the
*generator*, not about retrieval. **With an extractive generator, the retriever
barely matters.** Retrieval quality only becomes visible in the answers once the
generator is good enough to be misled by a bad context. Which is exactly what
Phase 3 is for: the measurement here is already in place, and it will start
moving when the generator stops being a copier.

---

## Known limitations

These are real, measured, and deliberately left in place. Fixing them is what
the later phases are for.

**1. One unanswerable case cannot be rejected by any retriever.**
Case 17 asks *"How many sick leave days do employees receive?"*. The word
`sick` appears nowhere in the knowledge base, but `leave`, `days`, `employees`
and `receive` all do — so the retriever confidently returns the annual leave
sentence and the generator answers "18". Distinguishing *"how many leave days"*
from *"how many sick leave days"* requires understanding that `sick` modifies
the answer, not understanding vocabulary alone.

Phase 2 measured this rather than assuming it, and the answer is that **no
retriever solves it**. In each case the unanswerable question scores *higher*
than a legitimate answerable one, so no threshold can separate them:

| Retriever | Case 17 (*"sick leave"*) | Lowest answerable case | Separable? |
| --- | --- | --- | --- |
| `lexical` | 0.589 | 0.369 (case 4) | No |
| `dense` | 0.479 | 0.396 (case 9) | No |
| `hybrid` | 0.534 | 0.462 (case 4) | No |

The interval is interleaved in all three, so this is not a threshold that needs
tuning. This is the clearest argument for moving to a language model generator,
and it survives the switch to embeddings.

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

**4. The relevance floors are calibrated on 21 cases, and their windows are
narrow.** Each floor sits in the middle of the range in which every answerable
case still finds its answer, and those ranges are only about 0.15 wide. On a
larger or different dataset the floors will have to be re-measured, and the
right way to do that is to sweep the floor and watch `hit_at_k`,
`answerable_rejection_rate` and `unanswerable_rejection_rate` together. The
framework will not stop a badly chosen floor from losing answers — it will
just make the loss visible.

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
| **2** ✅ | Dense retrieval (Ollama embeddings) and a hybrid retriever, with the same `Retriever` interface |
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

177 tests across twelve modules. They cover the tokenizer and stemmer, chunking
and BM25 ranking, every metric, dataset validation and the legacy format,
configuration overrides and the no-secrets rule, generator abstention, the
response cache, the embedding client, the dense and hybrid retrievers, the
retriever comparison, and the runner end to end including the report files it
writes.

**The suite needs no model server.** The embedding HTTP layer is replaced and
the dense and hybrid retrievers are driven by a fake embedder, so `make test`
passes on a machine where Ollama is not installed at all.

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

**`Could not reach Ollama at http://localhost:11434/v1`**

The dense and hybrid retrievers need a running Ollama server with an embedding
model:

```bash
ollama serve
ollama pull embeddinggemma
```

The error message says which of the two is wrong. If the server is up but the
model is missing, the response is a 404 and the message names the model to
pull. The lexical retriever needs neither, and the whole test suite runs
without a server at all.

---

## Author

Vidhya A
