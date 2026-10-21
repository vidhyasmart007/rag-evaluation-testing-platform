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
make compare   # compare the three retrievers
make compare-generators MODEL=qwen3:1.7b   # compare the generators
make install   # editable install, with a macOS fix (see Troubleshooting)
```

The evaluation runs with the extractive generator by default, so it needs
nothing but Python. To run it against a language model:

```bash
ollama pull qwen3:1.7b
PYTHONPATH=src python -m rageval.cli run \
  --set generation.provider=ollama \
  --set generation.model=qwen3:1.7b
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
│   ├── compare.py              # runs the arms of a comparison and tabulates them
│   ├── retrieval/
│   │   ├── base.py             # Retriever protocol
│   │   ├── lexical.py          # sentence splitting, chunking, BM25
│   │   ├── embeddings.py       # Ollama embeddings client, disk memoisation
│   │   ├── dense.py            # cosine similarity over embedded chunks
│   │   └── hybrid.py           # lexical and dense scores blended
│   ├── generation/
│   │   ├── base.py             # Generator protocol, GenerationResult
│   │   ├── extractive.py       # baseline generator
│   │   ├── ollama.py           # language model generator
│   │   └── cached.py           # disk memoisation for any generator
│   └── metrics/
│       ├── answer.py           # token_f1, exact_match, faithfulness, ...
│       └── retrieval.py        # hit@k, recall@k, precision@k, MRR
├── tests/                      # 226 tests
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
PYTHONPATH=src python -m rageval.compare --axis generator --model qwen3:1.7b
```

| Command | What it does |
| --- | --- |
| `run` | Evaluate the whole dataset and write a report |
| `cases` | List the dataset cases and any loader warnings |
| `config` | Print the effective configuration as JSON |
| `retrieve` | Show what the retriever returns for one question |
| `rageval.compare` | Run all three retrievers on the same dataset and print the metrics that differ |
| `rageval.compare --axis generator` | Compare the extractive baseline against the language model |

`rageval.compare` writes its reports to a temporary directory, so running a
comparison never clobbers the baseline in `reports/`.

The generator comparison turns the cache **off**, because a cached answer is a
lookup rather than a generation, and a comparison that reported a lookup time
as the cost of an arm would be measuring the cache instead of the model.

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
the retrieved context, the model name and **every setting that changes the
answer**. For the language model that means the temperature, the seed, the
token limit and the reasoning mode; for the extractive generator it means the
abstention threshold. A cached answer made at one temperature is not an answer
to the question asked at another, and serving it anyway would have the run
report the old number as if it were the new one.

Re-running the same evaluation serves every answer from disk instead of
regenerating it, which matters once the generator is a hosted model with a
per-day request cap.

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

## Generators

Two generators share one interface, so they are interchangeable:

```python
generate(question, context) -> GenerationResult
```

| Generator | How it answers | `generation.provider` |
| --- | --- | --- |
| `extractive` | Copies the sentence from the context with the highest word overlap with the question | `extractive` |
| `ollama` | Asks a local language model, and lets it refuse | `ollama` |

The extractive generator is not a placeholder. It is the arm every other
generator has to beat, and keeping it in the comparison is what makes the
numbers mean something.

### The language model generator

```bash
ollama serve
ollama pull qwen3:1.7b            # 1.4 GB

PYTHONPATH=src python -m rageval.cli run \
  --set generation.provider=ollama \
  --set generation.model=qwen3:1.7b
```

The model is called through Ollama's native chat API at `/api/chat`, with the
standard library's `urllib`. The project keeps **zero runtime dependencies**,
and one POST with a JSON body is not worth a dependency.

The native endpoint is used rather than the OpenAI-compatible one at
`/v1/chat/completions` for one concrete reason: **only the native endpoint
honours the `think` parameter**, and that parameter decides whether a
reasoning model spends a hundred tokens thinking before it answers. On Qwen3
the difference is not marginal — with reasoning on, a one-sentence answer
arrives in about 130 generated tokens and takes three times as long; with it
off the same answer takes 19 tokens.

```bash
--set generation.think=false       # fast: skip the reasoning mode
--set generation.think=true        # slow: let the model reason first
```

The parameter is only sent when it is set, because it is a parameter that only
some models accept.

**Abstention is the model's decision, not the framework's.** The prompt asks for
a small JSON object, `{"answer": ..., "abstained": ...}`, and the model sets
the flag itself. A guard rail applied around the generator would measure the
guard rail rather than the model, so `generation.min_answer_score` stays
extractive-only.

Three things are handled defensively, because a 1.7B model will not always
comply:

- **An empty context never reaches the model.** If retrieval found nothing,
  there is nothing to ground an answer in, and a model call would only buy a
  refusal, more slowly.
- **The response is parsed in three passes.** A strict JSON parse, then a JSON
  object embedded in prose or fenced in a code block, then a plain-text
  fallback.
- **A refusal written as the answer is still a refusal.** If the model says
  `"abstained": false` and then writes *"I don't know"*, the refusal wins,
  because the text is the behaviour that matters.

Every abstention records **why**: `empty_context` (nothing was retrieved),
`model_refusal` (the model declined), `empty_answer` (the model returned
nothing usable), or `below_min_answer_score` (the extractive threshold). A
framework that only records the boolean cannot tell a retrieval failure from a
generation failure.

### Reproducibility, measured

`temperature` is `0` and `seed` is `42` by default, and both are part of the
cache key. Over **19 independent runs** of the full dataset, 17 produced
byte-identical answers and identical metrics. In 2 of the 19, one answer —
case 13, *"How is code reviewed before it is merged?"* — gained a trailing
clause, and `token_f1` moved from 0.7973 to 0.7902.

So the seed gets you *near*-reproducibility, not exact reproducibility. The
seed fixes the sampling, not the GPU arithmetic: the Metal kernels are not
bit-deterministic, so two nearly tied tokens can flip. The response cache is
what makes a run exactly reproducible, which is why it is on by default.

The reports in `reports/` are one run each, chosen as the median of three so
that the table below and the evidence on disk agree.

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
    "min_answer_score": 0.45,
    "think": null,
    "timeout_seconds": 180.0
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
- **`generation.min_answer_score`** (0.45) is the **abstention floor**, and it
  applies to the extractive generator only. The generator refuses to answer when
  no sentence in the context covers enough of the question's content words. The
  language model generator decides for itself, so this knob does not apply to
  it.
- **`generation.temperature` and `generation.seed`** are the reproducibility
  settings for the language model. `0` and `42` by default. Both are part of
  the cache key.
- **`generation.think`** is `null` by default, which means the parameter is not
  sent at all. Set it to `false` for a fast run on a reasoning model, or to
  `true` to let the model reason before answering.
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

### The retrievers

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

### The generators

The same 21 cases, the lexical retriever, `qwen3:1.7b`:

| Metric | `extractive` | `ollama` (no reasoning) | `ollama` (reasoning) |
| --- | --- | --- | --- |
| `answer_correct` | **0.9412** | 0.8824 | 0.8824 |
| `token_recall` | **0.9765** | 0.9608 | 0.9216 |
| `token_f1` | 0.6097 | **0.7973** | 0.7653 |
| `faithfulness` | **0.8571** | 0.8452 | 0.7946 |
| `answer_relevance` | **0.7841** | 0.6413 | 0.5556 |
| `abstention_correct` | 0.9524 | 0.9524 | **1.0000** |
| `unanswerable_abstention_rate` | 0.7500 | 0.7500 | **1.0000** |
| `hallucination_rate` | 0.2500 | 0.2500 | **0.0000** |
| `mean_latency_ms` | 0 | 733 | 3568 |

Reproduce it with `make compare-generators MODEL=qwen3:1.7b`. The saved runs
are `reports/phase3_llm_lexical.json` (no reasoning),
`reports/phase3_llm_lexical_think.json` (reasoning) and
`reports/phase3_llm_hybrid.json`.

### What the generator comparison actually shows

**The language model writes much better answers.** `token_f1` jumps from 0.6097
to 0.7973 — a 31% improvement — and `exact_match` from 0.1176 to 0.3529. The
extractive generator can only ever copy one sentence, so its answers match the
reference wording by accident. The model answers in the reference's own
register.

**The extractive generator's `answer_correct` lead is an artefact of the
metric, not of the answers.** The headline score says the copier wins, 0.9412
against 0.8824. It does not. Two of the model's three "mistakes" are plainly
correct answers that the token rule marks wrong because of wording:

| Case | Reference | The model's answer | `token_recall` | Verdict |
| --- | --- | --- | --- | --- |
| 4 | *"Bengaluru, Karnataka."* | *"Bengaluru"* | 0.500 | Correct. Too terse. |
| 20 | *"The standard work week runs Monday through Friday."* | *"The standard work week is Monday through Friday."* | 0.833 | Correct. One word short. |
| 3 | *"The company uses Python as its primary programming language."* | *"Python"* | 0.167 | Correct. Far too terse. |

The rule is `token_recall >= 0.9`. *"Bengaluru"* is the right answer to
*"Which city is the office in?"* and it scores 0.5. This is a measured
argument for the judge phase, not a hunch: the token rule punishes a correct
answer for being concise and rewards a correct answer for being literal.

**The reasoning mode is what fixes the hallucination.** Without it, the model
answers case 17 (*"How many sick leave days do employees receive?"*) with
*"Employees receive 18 annual leave days per calendar year."* — the same
hallucination the extractive generator produces, because the context it was
handed says "leave" and not "sick". With the reasoning mode on, it refuses,
and `hallucination_rate` goes from 0.2500 to **0.0000**: all four unanswerable
cases are refused, including the one that no retriever can reject. The cost is
5× the latency and a slightly less faithful style.

**Faithfulness stopped being degenerate.** With a copying generator the metric
was almost always 1.0, because the answer *was* the context. With a
paraphrasing generator it reports 0.85 and 0.79, and it now measures
something: how much of the model's wording is actually supported.

**The retriever finally matters.** Phase 2 found that all three retrievers
produced byte-identical answers, because a copier is insensitive to its
context. With the language model, switching from lexical to hybrid changes
three of the 21 answers and moves `token_f1` from 0.7973 to 0.8136. The
retrieval work from Phase 2 was not wasted; it just had nothing to show until
the generator stopped being a copier.

### What the retriever comparison shows

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

**With an extractive generator, the end-to-end answers are identical across all
three retrievers.** Every metric about the *answer* is unchanged. That is not a
rounding artefact: the retriever picks a different top chunk in 4 of the 21
cases, but the extractive generator then picks the same sentence out of the
context either way, because the chunks overlap by a sentence and the generator
chooses on word coverage.

This is the most useful thing Phase 2 found, and it is a result about the
*generator*, not about retrieval. **With an extractive generator, the retriever
barely matters.** Phase 3 confirmed the prediction: with the language model
generator, the same switch from lexical to hybrid changes three answers and
moves `token_f1` from 0.7973 to 0.8136.

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
tuning.

**Phase 3 closed this one — but not with a threshold.** A language model with
its reasoning mode on *reads* the question and notices that the context talks
about annual leave, not sick leave, and refuses. `hallucination_rate` drops to
0.0000. It is not a retrieval fix: the retriever still hands over the same
confusing context, and the fix lives entirely in the generator. That is the
honest version of the result — the failure was never in the ranking, it was in
reading.

**2. The extractive baseline can only return one sentence.**
Case 8 (*"What is Retrieval-Augmented Generation?"*) has a reference answer that
spans two sentences of the source document. A generator that returns exactly
one sentence scores `token_recall = 0.6` and is marked incorrect. The answer is
not wrong, it is truncated. The language model answers this case correctly —
it is the one case where the copier loses outright.

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

**5. The answer correctness rule is a proxy — and now it is measurably wrong.**
`token_recall >= 0.9` is a reasonable stand-in for "says the same thing", but
it is not understanding. Phase 3 turned this from a worry into a measurement:
on the language model's answers the rule marks *"Bengaluru"* wrong for
*"Which city is the office in?"* (token recall 0.500) and *"The standard work
week is Monday through Friday."* wrong for a reference that reads *"The standard
work week runs Monday through Friday."* (token recall 0.833). Two of the
model's three "errors" are correct answers.

This is the strongest argument yet for the judge phase, and it comes with a
caveat: a judge is not automatically better. It has to be validated against
hand labels with Cohen's κ before any of its numbers are trusted, and until
that is done the token rule stays the headline and the judge is reported
alongside it.

**6. The seed does not make the model bit-reproducible.** Over 19 independent
runs, 17 produced identical answers and 2 differed on a single case. The seed
fixes the sampling, not the GPU arithmetic. The response cache is what makes a
run exactly reproducible.

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
| **3** ✅ | LLM answer generation via Ollama, seeded for reproducibility, with the extractive baseline kept as a comparison arm |
| **4** | LLM-as-judge on a different model family, validated against hand labels with Cohen's κ before it is trusted |
| **5** | Streamlit dashboard, then a public deployment |

The dashboard comes last on purpose. A UI over broken metrics is a UI over
broken metrics.

Phase 4 has its work cut out for it. The token rule now demonstrably marks
correct answers wrong, so the judge's first job is to be *right* about those
cases, and its second is to say so out loud if it is not.

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

226 tests across thirteen modules. They cover the tokenizer and stemmer, chunking
and BM25 ranking, every metric, dataset validation and the legacy format,
configuration overrides and the no-secrets rule, generator abstention, the
response cache, the embedding client, the dense and hybrid retrievers, the
language model generator (request shape, JSON parsing, abstention, timeouts and
HTTP failures), the retriever and generator comparisons, and the runner end to
end including the report files it writes.

**The suite needs no model server.** The HTTP layers of both the embedding
client and the language model generator are replaced, and the dense and hybrid
retrievers are driven by a fake embedder, so `make test` passes on a machine
where Ollama is not installed at all.

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

**`Could not reach Ollama at http://localhost:11434`**

The dense and hybrid retrievers and the language model generator need a running
Ollama server:

```bash
ollama serve
ollama pull embeddinggemma     # for the dense and hybrid retrievers
ollama pull qwen3:1.7b         # for the language model generator
```

The error message says which of the two is wrong. If the server is up but the
model is missing, the response is a 404 and the message names the model to
pull. The lexical retriever and the extractive generator need neither, and the
whole test suite runs without a server at all.

**The language model answers but it is slow**

A reasoning model such as Qwen3 spends about a hundred tokens thinking before
it answers, which is roughly five times the latency. `--set
generation.think=false` turns the reasoning mode off. The trade is measured:
without it the model hallucinates on case 17, with it `hallucination_rate` is
0.0000.

---

## Author

Vidhya A
