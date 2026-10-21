# RAG Evaluation and Testing Platform

A Python-based prototype for evaluating document retrieval and
answer quality in Retrieval-Augmented Generation (RAG) workflows.

## Project Overview

This project demonstrates a baseline evaluation workflow that
loads a knowledge base, retrieves relevant text chunks, selects
an extractive answer, calculates evaluation scores, and exports
results to CSV.

## Features

- Document loading and text chunking
- Lexical retrieval using word-overlap scoring
- Extractive answer selection
- Answer relevance scoring
- Lexical faithfulness approximation
- CSV evaluation reports
- Automated unit tests using pytest

## Project Structure

```text
LLM-EVALUATION-FRAMEWORK/
├── data/
│   ├── eval_dataset.json
│   └── knowledge_base.txt
├── reports/
│   ├── evaluation_report.csv
│   └── rag_evaluation_report.csv
├── src/
│   ├── evaluator.py
│   ├── rag_pipeline.py
│   └── retriever.py
├── tests/
│   └── test_evaluator.py
├── Day1.py
├── day2.py
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- JSON
- CSV
- pytest
- Git and GitHub

## Setup

Clone the repository and navigate into the project directory.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the testing dependency:

```bash
python -m pip install pytest
```

## Run the RAG Pipeline

```bash
python src/rag_pipeline.py
```

Enter a question about the sample knowledge base when prompted.

The pipeline retrieves text chunks, selects an answer from the
retrieved context, calculates baseline evaluation scores, and
writes the results to:

`reports/rag_evaluation_report.csv`

## Run Automated Tests

```bash
python -m pytest -v
```

The current unit tests cover basic answer relevance, lexical
faithfulness behavior, and empty retrieval handling.

## Evaluation Methodology

### Retrieval Relevance

Uses word-overlap scoring to rank document chunks against a
question.

### Answer Relevance

Uses lexical overlap between the question and selected answer.

### Faithfulness Approximation

Measures the proportion of answer terms that also appear in the
retrieved context.

These are simple baseline heuristics, not validated semantic
evaluation metrics.

## Limitations

- Retrieval uses lexical matching rather than semantic embeddings.
- Answer selection is extractive rather than LLM-generated.
- Lexical overlap cannot reliably detect contradictions or
  determine whether an answer is semantically supported.
- The current unit tests do not establish production readiness.
- No external LLM or embedding model is integrated yet.

## Future Improvements

- Integrate embedding-based semantic retrieval.
- Add an LLM-based answer-generation component.
- Evaluate retrieval precision, recall, and ranking quality.
- Add reference-based answer evaluation.
- Expand automated tests with edge cases and integration tests.
- Add a larger, labeled evaluation dataset.

## Author

Vidhya A