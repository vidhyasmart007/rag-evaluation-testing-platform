# Convenience targets for the evaluation platform.
#
# The project uses a src layout, so the package is imported from ./src.
# Every target sets PYTHONPATH accordingly, which means the commands work
# whether or not the package has been installed into the environment.

VENV := .venv
VENV_PY := $(VENV)/bin/python
PYTHON := $(shell test -x $(VENV_PY) && echo $(VENV_PY) || command -v python3)

SRC := src
PYTHONPATH := $(SRC)
export PYTHONPATH

# The judge used by the validate-judge target. The local one needs no key.
JUDGE_PROVIDER ?= ollama
JUDGE_MODEL ?= llama3.2:3b

.PHONY: help install run test cases config retrieve compare compare-generators \
	validate-judge lint clean

help:
	@echo "Targets:"
	@echo "  install   install the package and its development extras"
	@echo "  run       run the evaluation suite and write a report"
	@echo "  test      run the test suite"
	@echo "  cases     list the dataset cases"
	@echo "  config    print the effective configuration"
	@echo "  retrieve  show what the retriever returns for a question"
	@echo "  compare   compare the lexical, dense and hybrid retrievers"
	@echo "  compare-generators  compare the extractive baseline against a model"
	@echo "  validate-judge  grade the hand labels and measure the judge"
	@echo "  lint      run ruff and mypy"
	@echo "  clean     remove caches and generated reports"

install:
	$(PYTHON) -m pip install -e ".[dev]"
	@# macOS: the hidden file attribute is inherited by everything created
	@# inside a hidden directory, and Python 3.13 skips any .pth file that
	@# carries it. That silently breaks editable installs, so clear it.
	-@chflags -R nohidden "$(VENV)" 2>/dev/null
	-@for d in $(VENV)/lib/python*/site-packages; do \
		chflags nohidden "$$d" "$$d"/*.pth 2>/dev/null || true; \
	done

run:
	$(PYTHON) -m rageval.cli run

test:
	$(PYTHON) -m pytest

cases:
	$(PYTHON) -m rageval.cli cases

config:
	$(PYTHON) -m rageval.cli config

retrieve:
	$(PYTHON) -m rageval.cli retrieve "$(QUESTION)"

compare:
	$(PYTHON) -m rageval.compare

compare-generators:
	$(PYTHON) -m rageval.compare --axis generator --model $(MODEL)

validate-judge:
	$(PYTHON) -m rageval.cli validate-judge \
		--labels data/hand_labels.json \
		--set judge.provider=$(JUDGE_PROVIDER) \
		--set judge.model=$(JUDGE_MODEL) \
		--set judge.think=false

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .cache
	find . -name '__pycache__' -type d -not -path './.git/*' -exec rm -rf {} +
	rm -f reports/run_*.json reports/run_*.csv
