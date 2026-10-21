import json
import csv
import re
from pathlib import Path


# Project paths
DATASET_PATH = Path("data/eval_dataset.json")
REPORT_DIR = Path("reports")
REPORT_PATH = REPORT_DIR / "evaluation_report.csv"


def normalize_text(text):
    """Normalize text before comparing answers."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def evaluate_answer(expected, actual):
    """Compare expected and actual answers."""

    if not isinstance(expected, str) or not isinstance(actual, str):
        return "INVALID"

    if not expected.strip() or not actual.strip():
        return "FAIL"

    expected_normalized = normalize_text(expected)
    actual_normalized = normalize_text(actual)

    if expected_normalized == actual_normalized:
        return "PASS"

    return "FAIL"


def load_dataset():
    """Load and validate the evaluation dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a JSON list.")

    required_fields = {
        "id",
        "question",
        "expected_answer",
        "actual_answer"
    }

    for index, case in enumerate(dataset):
        if not isinstance(case, dict):
            raise ValueError(
                f"Test case at index {index} must be an object."
            )

        missing_fields = required_fields - case.keys()

        if missing_fields:
            raise ValueError(
                f"Test case at index {index} is missing: "
                f"{', '.join(sorted(missing_fields))}"
            )

    return dataset


def generate_report(results):
    """Export test results to a CSV report."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_id",
        "question",
        "expected_answer",
        "actual_answer",
        "result"
    ]

    with REPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nReport saved to: {REPORT_PATH}")


def main():
    dataset = load_dataset()
    results = []

    for case in dataset:
        result = evaluate_answer(
            case["expected_answer"],
            case["actual_answer"]
        )

        results.append({
            "test_id": case["id"],
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "actual_answer": case["actual_answer"],
            "result": result
        })

        print(f"Test ID: {case['id']}")
        print(f"Question: {case['question']}")
        print(f"Evaluation: {result}")
        print("-" * 40)

    passed = sum(r["result"] == "PASS" for r in results)
    failed = sum(r["result"] == "FAIL" for r in results)
    invalid = sum(r["result"] == "INVALID" for r in results)
    total = len(results)

    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total test cases: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Invalid: {invalid}")

    if total:
        pass_rate = passed / total * 100
        print(f"Pass rate: {pass_rate:.2f}%")
    else:
        print("No test cases available.")

    generate_report(results)


if __name__ == "__main__":
    main()