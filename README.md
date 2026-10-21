# RAG Evaluation and Testing Platform

A Python-based prototype for testing document retrieval and evaluating answer quality in a Retrieval-Augmented Generation (RAG) workflow.

## Project Overview

This project demonstrates a baseline approach to evaluating a document-based question-answering pipeline. It retrieves relevant text from a knowledge base, selects an answer from the retrieved context, calculates evaluation scores, and generates CSV reports.

## Features

- Document loading and text chunking
- Lexical retrieval using word-overlap scoring
- Extractive answer selection
- Basic answer relevance evaluation
- Lexical faithfulness approximation
- Retrieval relevance scoring
- CSV evaluation reports
- Automated unit tests using pytest

## Technology Stack

- Python
- JSON for structured evaluation datasets
- CSV for evaluation reports
- pytest for automated testing
- Git and GitHub for version control

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

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/vidhyasmart007/rag-evaluation-testing-platform.git
cd rag-evaluation-testing-platform
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the testing dependency

```bash
python -m pip install pytest
```

The retrieval and evaluation modules use Python's standard library. No external LLM API key is required for the current prototype.

## Run the RAG Pipeline

```bash
python src/rag_pipeline.py
```

Enter a question about the sample knowledge base when prompted.

The pipeline retrieves relevant text chunks, selects an extractive answer, calculates baseline evaluation scores, and saves the results to:

`reports/rag_evaluation_report.csv`

## Run the Evaluation Script

```bash
python day2.py
```

This evaluates the predefined dataset in `data/eval_dataset.json` and generates:

`reports/evaluation_report.csv`

## Run Automated Tests

```bash
python -m pytest -v
```

The initial test suite contains four tests covering basic answer relevance, lexical faithfulness behavior, and empty retrieval handling.

Run the tests yourself to see the current results.

## Evaluation Metrics

### Retrieval Relevance

Uses word overlap between a question and retrieved text chunks to calculate a baseline relevance score.

### Answer Relevance

Measures lexical overlap between the question and the selected answer.

### Faithfulness Approximation

Measures how many answer terms also appear in the retrieved context.

These metrics are simple heuristics and should not be interpreted as validated measures of semantic accuracy.

## Limitations

- Retrieval uses lexical matching rather than semantic embeddings.
- Answer selection is extractive rather than LLM-generated.
- Word overlap cannot reliably detect contradictions or establish semantic equivalence.
- The current test suite provides basic functional checks, not comprehensive validation.
- The prototype does not currently integrate an external LLM or embedding model.

## Future Improvements

- Add embedding-based semantic retrieval.
- Integrate an LLM for answer generation.
- Evaluate retrieval precision, recall, and ranking quality.
- Add reference-based answer evaluation.
- Expand the test suite with integration and edge-case tests.
- Explore established RAG evaluation frameworks such as Ragas or DeepEval.

## Author

Vidhya A