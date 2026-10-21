"""Filesystem locations resolved relative to the repository root.

Every module resolves paths through this file so the project behaves the same
no matter which directory a command is launched from.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
ROOT = SRC_DIR.parent

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
CACHE_DIR = ROOT / ".cache"

DEFAULT_DATASET_PATH = DATA_DIR / "eval_dataset.json"
DEFAULT_KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.txt"
